import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import json

# 개인화 모듈 import
sys.path.extend(['D:\\_python@local\\ShortPunchTrader'])
import _custom_module_32 as cm
import API_kakao
import lotto_logic

# 에러 발생 시 파일로 출력
# folder_log = 'C:\\Users\\chury\\iCloudDrive\\python_log'
# sys.stderr = open(file=os.path.join(folder_log, 'error_lotto_anal.log'), mode='wt', encoding='utf-8')


# noinspection PyPep8Naming,PyProtectedMember
class LottoAnal:
    def __init__(self, b_test=False, s_date=None, n_range=None, b_tqdm=True):
        # config.json 불러오기
        with open('config.json', 'rt', encoding='cp949') as file:
            dic_config = json.load(file)

        # 폴더 정의
        self.folder = str(dic_config['folder_work'])
        # self.folder = os.getcwd()
        self.folder_history = os.path.join(self.folder, 'history')
        self.folder_run = os.path.join(self.folder, 'run')
        self.folder_result = os.path.join(self.folder, 'result')
        self.folder_result_확률예측 = os.path.join(self.folder_result, '확률예측')
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.folder_history, exist_ok=True)
        os.makedirs(self.folder_run, exist_ok=True)
        os.makedirs(self.folder_result, exist_ok=True)
        os.makedirs(self.folder_result_확률예측, exist_ok=True)

        # path 정의
        self.path_args = os.path.join(self.folder_run, 'dic_args.pkl')

        # dic_args 정의
        if 'dic_args.pkl' in os.listdir(self.folder_run):
            self.dic_args = pd.read_pickle(self.path_args)
        else:
            self.dic_args = {}

        # self 변수 정의
        self.b_test = b_test
        self.b_tqdm = b_tqdm
        self.dt_지금 = pd.Timestamp('now')
        self.s_오늘날짜 = self.dt_지금.strftime('%Y%m%d')
        self.s_기준일자 = s_date if s_date is not None else self.s_오늘날짜
        self.n_학습차수 = n_range if n_range is not None else 500
        self.n_과거회차 = 10

        self.li_45개번호 = [f'{i + 1:02}' for i in range(45)]
        self.li_45개번호_2개조합 = self._번호조합_2개(li_전체번호=self.li_45개번호)
        self.li_6개순번_2개조합 = self._순번조합_2개()

        # log path 정의
        folder_log = str(dic_config['folder_log'])
        os.makedirs(folder_log, exist_ok=True)
        self.path_log = os.path.join(folder_log, f'log_lotto_anal_{self.s_오늘날짜}.log')
        # self.path_log = os.path.join('C:\\Users\\chury\\iCloudDrive\\python_log', f'log_lotto_anal_{self.s_오늘날짜}.log')

        # 인스턴스 불러오기
        self.t = cm.Tools()

        # 로그 기록
        self.t.make_log2(s_text='### Lotto 분석 시작 ###', li_loc=['console', 'file'], s_file=self.path_log)

    def 추첨이력_업데이트(self):
        """ 회차별 추첨이력 Update 후 history 폴더에 csv 저장 (네이버 크롤링) """
        # 로그 기록
        self.t.make_log2(s_text='# 추첨이력 업데이트 #', li_loc=['console', 'file'], s_file=self.path_log)

        # 저장된 데이터 확인
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), engine='python')

        # 기존 데이터 정보 확인
        df_이력 = df_이력.sort_values('date', ascending=False).reset_index(drop=True).astype(str)
        n_회차_최종 = int(df_이력['seq'][0])
        s_추첨일_최종 = df_이력['date'][0]
        dt_추첨일_최종 = pd.Timestamp(s_추첨일_최종)

        # 최종 데이터 이후 7일 경과 데이터 수집
        while self.dt_지금.date() > dt_추첨일_최종 + pd.Timedelta(days=7):
            # 신규 데이터 크롤링
            n_회차_최종 += 1
            df_조회 = self._추첨번호_크롤링(n_회차=n_회차_최종)

            # 이전 데이터와 병합
            df_이력 = pd.concat([df_이력, df_조회], axis=0)
            df_이력 = df_이력.drop_duplicates().sort_values('date', ascending=False).reset_index(drop=True)
            n_회차_최종 = int(df_이력['seq'][0])
            s_추첨일_최종 = df_이력['date'][0]
            dt_추첨일_최종 = pd.Timestamp(s_추첨일_최종)

            # 완료 후 csv 저장
            df_이력.to_csv(os.path.join(self.folder_history, 'lotto_history.csv'), index=False, encoding='cp949')

            # 로그 기록
            self.t.make_log2(s_text=f"{n_회차_최종}회차 정보 업데이트 완료 - {s_추첨일_최종}",
                             li_loc=['console', 'file'], s_file=self.path_log)

        # 신규 데이터 미존재 시 메세지 출력
        if self.dt_지금.date() <= dt_추첨일_최종 + pd.Timedelta(days=7):
            # 로그 기록
            self.t.make_log2(s_text=f"신규 데이터 미존재 - 마지막 데이터 {n_회차_최종}회차 / {s_추첨일_최종}",
                             li_loc=['console', 'file'], s_file=self.path_log)

    def 전처리_데이터변환(self):
        """ 추첨이력 데이터 변환 (one-hot encoding, 2개 조합, 과거 데이터 생성) """
        # 로그 기록
        self.t.make_log2(s_text='# 데이터 전처리 #', li_loc=['console', 'file'], s_file=self.path_log)

        # 데이터 불러오기
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), engine='python').astype(str)

        # 데이터 정렬 (일자순)
        df_이력 = df_이력.sort_values('date').reset_index(drop=True)

        # 당첨번호 2개 조합 데이터 생성
        df_이력 = self._당첨번호_2개조합(df=df_이력)

        # 6개 당첨번호 one-hot encoding
        df_이력 = self._ohe_당첨번호(df=df_이력)

        # 과거 회차 당첨번호 정보 추가
        df_이력 = self._과거회차생성(df=df_이력, n_기간=self.n_과거회차)

        # 당첨번호 2개 조합 one-hot encoding
        df_이력 = self._ohe_당첨번호_2개조합(df=df_이력)

        # 과거 회차 당첨번호 정보 추가 (2개 조합)
        df_이력 = self._과거회차생성_2개조합(df=df_이력, n_기간=self.n_과거회차)

        # NaN 삭제
        df_이력 = df_이력.dropna().reset_index(drop=True)

        # df 저장
        self.dic_args['df_이력'] = df_이력
        pd.to_pickle(self.dic_args, self.path_args)

    def 분석모델_생성(self):
        """ x, y 데이터셋 생성 및 Random Forest 학습 모델 생성 """
        # 로그 기록
        self.t.make_log2(s_text=f'# 분석모델 생성_{self.s_기준일자} #', li_loc=['console', 'file'], s_file=self.path_log)

        # 전처리 데이터 불러오기
        df_이력 = self.dic_args['df_이력']

        # 검증구간 데이터 잘라내기
        s_기준일자 = pd.Timestamp(self.s_기준일자).strftime('%Y.%m.%d')
        df_데이터 = df_이력[df_이력['date'] <= s_기준일자].copy()
        df_데이터 = df_데이터[(-1 * self.n_학습차수):]

        # x 데이터셋 생성
        li_컬럼명_x = [s for s in df_데이터.columns if s.startswith('p')]
        ary_x_학습 = self._get_x(df=df_데이터, li_col_x=li_컬럼명_x)

        # 모델 저장용 dic 생성
        dic_models = dict()

        # 45개 번호별 모델 생성
        s_대상 = tqdm(self.li_45개번호, desc='45개 번호별 학습') if self.b_tqdm else self.li_45개번호
        for s_번호 in s_대상:
            ary_y_학습 = self._get_y(df=df_데이터, s_col_y=f'no{s_번호}')

            # random forest 모델 학습
            model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=99)
            model.fit(ary_x_학습, ary_y_학습)

            # 모델 저장
            dic_models[f'model_no{s_번호}'] = model

        # 2개 조합 번호별 모델 생성
        s_대상 = tqdm(self.li_45개번호_2개조합, desc='2개 조합별 학습') if self.b_tqdm else self.li_45개번호_2개조합
        for s_번호_조합 in s_대상:
            ary_y_학습 = self._get_y(df=df_데이터, s_col_y=f'no{s_번호_조합}')

            # random forest 모델 학습
            model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=99)
            model.fit(ary_x_학습, ary_y_학습)

            # 모델 저장
            dic_models[f'model_no{s_번호_조합}'] = model

        # model 정보 저장
        self.dic_args['df_데이터'] = df_데이터
        self.dic_args['dic_models'] = dic_models

        # dic_args 저장
        pd.to_pickle(self.dic_args, self.path_args)

    def 확률계산_1개2개조합(self):
        """ 분석모델을 통해 번호별 확률 계산하여 result 폴더에 csv 저장 """
        # 로그 기록
        self.t.make_log2(s_text='# 번호별 확률 계산 (1개, 2개 조합) #', li_loc=['console', 'file'], s_file=self.path_log)

        # 예측용 데이터 준비
        df_이력 = self.dic_args['df_이력']
        s_기준일자 = pd.Timestamp(self.s_기준일자).strftime('%Y.%m.%d')
        df_데이터 = df_이력[df_이력['date'] <= s_기준일자]
        df_데이터 = df_데이터[-1:]

        # 예측 기준정보 생성
        n_예측_차수, s_예측_추첨일 = self._예측기준정보생성(s_기준일자=self.s_기준일자)
        dic_예측_기준정보 = dict()
        dic_예측_기준정보['n_예측_차수'] = n_예측_차수
        dic_예측_기준정보['s_예측_추첨일'] = s_예측_추첨일

        # 예측용 x 데이터셋 생성
        li_컬럼명_제외 = (['seq', 'date']
                     + [s for s in df_데이터.columns if s.startswith('win')]
                     + [s for s in df_데이터.columns if s.startswith(f'p{self.n_과거회차}')])
        li_컬럼명_x = [s for s in df_데이터.columns if s not in li_컬럼명_제외]
        ary_x_예측 = df_데이터.loc[:, li_컬럼명_x].values

        # 45개 번호 확률 예측
        df_확률_1개 = self._확률예측(ary_x=ary_x_예측, li_numbers=self.li_45개번호, dic_pred_info=dic_예측_기준정보)
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_1개.to_csv(os.path.join(self.folder_result_확률예측, s_파일명), index=False)

        # 2개 조합 번호 확률 예측
        df_확률_2개 = self._확률예측(ary_x=ary_x_예측, li_numbers=self.li_45개번호_2개조합, dic_pred_info=dic_예측_기준정보)
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_2개.to_csv(os.path.join(self.folder_result_확률예측, s_파일명), index=False)

        # 결과 저장
        self.dic_args['df_확률_1개'] = df_확률_1개
        self.dic_args['df_확률_2개'] = df_확률_2개
        self.dic_args['n_예측_차수'] = n_예측_차수
        self.dic_args['s_예측_추첨일'] = s_예측_추첨일

        # dic_args 저장
        pd.to_pickle(self.dic_args, self.path_args)

    def 확률계산_6개조합(self):
        """ 1개, 2개 확률 상위 n개씩 번호 조합하여 6개 번호의 확률 산출 후 result 폴더에 csv 저장 """
        # 예측 기준정보 생성
        n_예측_차수, s_예측_추첨일 = self._예측기준정보생성(s_기준일자=self.s_기준일자)

        # 로그 기록
        self.t.make_log2(s_text=f'# 번호 6개 조합별 확률 산출 ({n_예측_차수:,}차_{s_예측_추첨일}추첨) #',
                         li_loc=['console', 'file'], s_file=self.path_log)

        # 확률 분석 결과 가져오기
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_1개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_2개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
        self.dic_args['df_확률_1개'] = df_확률_1개
        self.dic_args['df_확률_2개'] = df_확률_2개

        # 확률 상위 n개 번호 가져오기 (unique)
        li_번호_확률상위n개 = self._확률상위n개(df1=df_확률_1개, df2=df_확률_2개, n_top=30)

        # 번호 조합 생성 (경우의 수 모두 구하기)
        df_번호조합 = self._번호조합(li_numbers=li_번호_확률상위n개)

        # 번호 조합별 확률값 산출
        df_번호조합확률 = self._번호조합확률(df_numbers=df_번호조합)

        # csv 저장
        s_파일명 = f'확률예측_6개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_번호조합확률.to_csv(os.path.join(self.folder_result_확률예측, s_파일명), index=False)

        # 결과 저장
        self.dic_args['df_번호조합확률'] = df_번호조합확률

        # dic_args 저장
        pd.to_pickle(self.dic_args, self.path_args)

    def 번호선정_5세트(self, s_선정로직):
        """ 최종 구매할 6개 번호 * 5개 세트 선정 후 result 폴더에 csv 저장 (기존 데이터는 csv로 읽어오기) """
        # 예측 기준정보 생성
        n_예측_차수, s_예측_추첨일 = self._예측기준정보생성(s_기준일자=self.s_기준일자)

        # 로그 기록
        self.t.make_log2(s_text=f'# {s_선정로직} ({n_예측_차수:,}차_{s_예측_추첨일}추첨) #',
                         li_loc=['console', 'file'], s_file=self.path_log)

        # 확률 분석 결과 가져오기
        dic_확률 = dict()
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_1개'] = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_2개'] = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
        s_파일명 = f'확률예측_6개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_6개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
        for n in range(6):
            df_확률_6개[f'no{n + 1}'] = df_확률_6개[f'no{n + 1}'].apply(lambda x: f'{x:02}')
        dic_확률['df_확률_6개'] = df_확률_6개

        # 번호 선정
        s_전략명 = None
        df_6개번호_5개세트 = None
        if s_선정로직 == '번호선정_2개번호_최빈수':
            s_전략명, df_6개번호_5개세트 = lotto_logic.번호선정로직_2개번호_최빈수(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_2개번호_최빈수_중복제외':
            s_전략명, df_6개번호_5개세트 = lotto_logic.번호선정로직_2개번호_최빈수_중복제외(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_2개번호_따라가기':
            s_전략명, df_6개번호_5개세트 = lotto_logic.번호선정로직_2개번호_따라가기(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_2개번호_따라가기_확률반영':
            s_전략명, df_6개번호_5개세트 = lotto_logic.번호선정로직_2개번호_따라가기_확률반영(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_복합로직_최빈수_따라가기':
            s_전략명, df_6개번호_5개세트 = lotto_logic.번호선정로직_복합로직_최빈수_따라가기(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_1개2개연계_최빈수연계':
            s_전략명, df_6개번호_5개세트 = lotto_logic.번호선정로직_1개2개연계_최빈수연계(dic_확률=dic_확률)

        # 결과 저장용 폴더 생성
        folder_result_전략 = os.path.join(self.folder_result, s_전략명)
        os.makedirs(folder_result_전략, exist_ok=True)

        # csv 저장
        s_파일명 = f'확률예측_번호선정_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_6개번호_5개세트.to_csv(os.path.join(folder_result_전략, s_파일명), index=False)

        # 결과 저장
        self.dic_args['df_6개번호_5개세트'] = df_6개번호_5개세트
        self.dic_args['folder_result_전략'] = folder_result_전략

        # dic_args 저장
        pd.to_pickle(self.dic_args, self.path_args)

    def 결과확인(self):
        """ history 데이터 사용하여 예측 결과 확인 후 result 폴더에 csv 저장 """
        # 예측 기준정보 생성
        # n_예측_차수, s_예측_추첨일 = self._예측기준정보생성(s_기준일자=self.s_기준일자)
        folder_result_전략 = self.dic_args['folder_result_전략']
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), engine='python').astype(str)
        df_이력 = df_이력.sort_values('date').reset_index(drop=True)

        # 번호예측 파일 확인
        li_files = [file for file in os.listdir(folder_result_전략) if '확률예측_번호선정_' in file and '.csv' in file]
        li_정보_전체 = [li_정보.split('_') for li_정보 in li_files]

        # 회차 추첨일 매칭 dict 생성
        df_회차_추첨일 = pd.DataFrame()
        df_회차_추첨일['n_회차'] = [int(li_정보[2].replace('차', '')) for li_정보 in li_정보_전체]
        df_회차_추첨일['s_추첨일'] = [li_정보[3].replace('추첨.csv', '') for li_정보 in li_정보_전체]
        df_회차_추첨일 = df_회차_추첨일.sort_values('n_회차')
        dic_회차_추첨일 = df_회차_추첨일.set_index('n_회차').to_dict()['s_추첨일']

        # 미확인 회차 생성
        li_결과파일 = [file for file in os.listdir(folder_result_전략) if '확률예측_추첨결과_' in file and '.csv' in file]
        li_결과차수 = [li_정보.split('_')[2] for li_정보 in li_결과파일]

        # 미확인 회차 결과 생성
        for n_회차 in df_회차_추첨일['n_회차'].values:
            if f'{n_회차}차' in li_결과차수:
                continue
            else:
                # 예측 차수, 추첨일 생성
                n_예측_차수 = n_회차
                s_예측_추첨일 = dic_회차_추첨일[n_예측_차수]

                # 로그 기록
                self.t.make_log2(s_text=f'# 예측 번호 결과 확인_이전 데이터 확인용 ({n_예측_차수:,}차_{s_예측_추첨일}추첨) #',
                                 li_loc=['console', 'file'], s_file=self.path_log)

                # 해당 차수 당첨번호 불러오기
                df_이력_차수 = df_이력[df_이력['seq'] == str(n_예측_차수)]
                if len(df_이력_차수) == 0:
                    self.t.make_log2(s_text=f'# 추첨 이력 미존재 ({n_예측_차수:,}차_{s_예측_추첨일}추첨) #',
                                     li_loc=['console', 'file'], s_file=self.path_log)
                    return
                ary_당첨번호 = df_이력_차수.loc[:, 'win_1': 'win_6'].values[0]

                # 선정 번호 가져오기
                s_파일 = f'확률예측_번호선정_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
                folder_result_전략 = self.dic_args['folder_result_전략']
                df_6개번호_5개세트 = pd.read_csv(os.path.join(folder_result_전략, s_파일), engine='python')
                df_6개번호_5개세트 = df_6개번호_5개세트.loc[:, 'no1': 'no6']
                for n in range(6):
                    df_6개번호_5개세트[f'no{n + 1}'] = df_6개번호_5개세트[f'no{n + 1}'].apply(lambda x: f'{x:02}')

                # 결과 확인
                li_당첨번호 = [int(s_번호) for s_번호 in ary_당첨번호]
                li_결과_전체 = []
                for ary_6개번호 in df_6개번호_5개세트.values:
                    li_당첨확인 = [1 if int(s_번호) in li_당첨번호 else 0 for s_번호 in ary_6개번호]
                    li_결과 = li_당첨확인 + [sum(li_당첨확인)] + list(ary_6개번호) + list(ary_당첨번호)
                    li_결과_전체.append(li_결과)

                # 결과 df 변화
                li_컬럼명 = [f'win{n + 1}' for n in range(6)] + ['cnt_win']\
                         + [f'no{n + 1}' for n in range(6)] + [f'win_{n + 1}' for n in range(6)]
                df_결과 = pd.DataFrame(li_결과_전체, columns=li_컬럼명)
                df_결과['seq'] = str(n_예측_차수)
                df_결과['date'] = pd.Timestamp(s_예측_추첨일).strftime('%Y.%m.%d')

                # 상금 입력
                df_결과['award'] = df_결과['cnt_win'].apply(lambda x: 0 if x <= 2 else 5000 if x == 3 else 50000 if x == 4
                else 1000000 if x == 5 else 1000000000)

                # 컬럼 순서 정리
                li_컬럼명 = ['seq', 'date'] + [f'win{n + 1}' for n in range(6)]\
                         + ['cnt_win', 'award'] + [f'no{n + 1}' for n in range(6)] + [f'win_{n + 1}' for n in range(6)]
                df_결과 = df_결과.loc[:, li_컬럼명]

                # csv 저장
                n_갯수 = df_결과['cnt_win'].max()
                n_상금 = df_결과['award'].sum()
                folder_result_전략 = self.dic_args['folder_result_전략']
                s_파일명 = f'확률예측_추첨결과_{n_예측_차수}차_{s_예측_추첨일}추첨_{n_갯수}개_{n_상금}원.csv'
                df_결과.to_csv(os.path.join(folder_result_전략, s_파일명), index=False)

                # 결과 저장
                self.dic_args['df_결과'] = df_결과

                # dic_args 저장
                pd.to_pickle(self.dic_args, self.path_args)


    def send_by_kakao(self):
        ''' 추출한 번호 set을 kakao api를 통해 전송 '''
        # 로그 기록
        self.t.make_log(s_text='### 카카오톡 전송 ###', li_loc=['console', 'file'], s_file=self.path_log)

        # 데이터 불러오기
        df = self.dic_args['df_set'].copy()

        # 변수 지정
        s_seq = df['seq'].values[-1]
        s_date = df['date'].values[-1]
        ary_set = df.loc[:, 'no1':'no6'].values

        # 텍스트 생성
        s_text1 = f"[회차] {s_seq} [추첨일] {s_date} (5/7)"
        s_text2 = f"[회차] {s_seq} [추첨일] {s_date} (7/7)"
        for i in range(len(ary_set)):
            ary_no = ary_set[i]
            s_no = f'\n게임{i + 1}# '
            for n_no in ary_no:
                s_no += f'{n_no:>2}, '

            if i + 1 <= 5:
                s_text1 += s_no[:-2]
            else:
                s_text2 += s_no[:-2]

        # 카카오톡으로 보내기
        k = API_kakao.kakaoAPIcontrol()

        for s_text in [s_text1, s_text2]:
            k.send_message(s_user='알림봇', s_friend='여봉이', s_text=s_text)
            # 로그 기록
            self.t.make_log(s_text=s_text, li_loc=['console', 'file'], s_file=self.path_log)

######################################################################################################################
    @staticmethod
    def _번호조합_2개(li_전체번호):
        """ 45개 번호를 2개씩 조합하여 li 리턴 """
        li_번호조합 = []

        li_번호1 = [int(x) for x in li_전체번호]
        for n_번호1 in li_번호1:
            li_번호2 = [x for x in li_번호1 if x > n_번호1]

            for n_번호2 in li_번호2:
                s_번호조합 = f'{n_번호1:02}|{n_번호2:02}'
                li_번호조합.append(s_번호조합)

        return li_번호조합

    @staticmethod
    def _순번조합_2개():
        """ 6개 순번을 2개씩 조합하여 15개짜리 li 리턴 """
        li_순번조합 = []

        li_순번1 = [n + 1 for n in range(6)]
        for n_순번1 in li_순번1:
            li_순번2 = [n for n in li_순번1 if n > n_순번1]

            for n_순번2 in li_순번2:
                s_순번조합 = f'{n_순번1}|{n_순번2}'
                li_순번조합.append(s_순번조합)

        return li_순번조합

    @staticmethod
    def _추첨번호_크롤링(n_회차):
        """ 네이버에서 추첨번호 크롤링하여 df 리턴 (조회 실패 시 'error' 리턴) """
        import requests
        from bs4 import BeautifulSoup

        # 조회 요청
        url = f'https://search.naver.com/search.naver?query={n_회차}회로또'
        response = requests.get(url)

        # 조회 성공 시 날짜, 번호 수집
        if response.status_code == 200:
            # 뷰티풀숩 설정
            soup = BeautifulSoup(response.text, 'html.parser')

            # 날짜 가져오기
            # tag_date = soup.select_one('#_lotto > div > div.lotto_tit > h3 > a > span')
            tag_date = soup.find('div', class_='select_tab')
            li_date = tag_date.text.split()
            s_추첨일 = li_date[1][1:-1]

            # 당첨번호 가져오기
            tag_numbers = soup.select('span.ball')
            li_추첨번호 = [tag_no.text for tag_no in tag_numbers]

        # 조회 실패 시 'error' 리턴
        else:
            s_추첨일 = 'error'
            li_추첨번호 = ['error'] * 7

        # df로 변환
        df = pd.DataFrame()
        df['seq'] = [str(n_회차)]
        df['date'] = s_추첨일
        df['win_1'] = li_추첨번호[0]
        df['win_2'] = li_추첨번호[1]
        df['win_3'] = li_추첨번호[2]
        df['win_4'] = li_추첨번호[3]
        df['win_5'] = li_추첨번호[4]
        df['win_6'] = li_추첨번호[5]
        df['win_bonus'] = li_추첨번호[6]

        return df

    def _당첨번호_2개조합(self, df):
        """ 당첨번호 6개를 2개씩 조합하여 컬럼 추가"""
        df = df.copy()

        # 순번 조합별 컬럼 생성
        for s_순번조합 in self.li_6개순번_2개조합:
            li_순번조합 = s_순번조합.split('|')
            s_순번1 = f'win_{li_순번조합[0]}'
            s_순번2 = f'win_{li_순번조합[1]}'
            s_순번조합 = f'win_{s_순번조합}'

            # 컬럼값 입력
            df[s_순번조합] = df[s_순번1].apply(lambda x: f'{int(x):02}') + '|' + df[s_순번2].apply(lambda x: f'{int(x):02}')

        return df

    def _ohe_당첨번호(self, df):
        """ 당첨번호 6개를 one-hot encoding 으로 변환 """
        df = df.copy()

        # 45개 번호에 대해 one-hot encoding 진행
        for s_번호_확인할거 in self.li_45개번호:
            df[f'no{s_번호_확인할거}'] = 0

            # 6개 당첨번호에 대해 해당 번호에 1 표기
            for n_순번 in range(1, 7):
                df.loc[df[f'win_{n_순번}'] == s_번호_확인할거, f'no{s_번호_확인할거}'] = 1

        return df

    def _ohe_당첨번호_2개조합(self, df):
        """ 당첨번호 6개를 2개씩 조합하여 one-hot encoding 으로 변환 """
        df = df.copy()

        # 45개 번호에 대한 2개 조합 번호에 대해 one-hot encoding 진행
        for s_번호_확인할거 in self.li_45개번호_2개조합:
            df[f'no{s_번호_확인할거}'] = 0

            # 순번 2개 조합(15개)에 대해 해당 번호에 1 표기
            for s_순번조합 in self.li_6개순번_2개조합:
                df.loc[df[f'win_{s_순번조합}'] == s_번호_확인할거, f'no{s_번호_확인할거}'] = 1

        return df

    def _과거회차생성(self, df, n_기간):
        """ 과거회차 당첨번호를 동일 열에 기입 """
        df = df.copy()

        # 과거회차별 컬럼 생성 (1개 번호)
        for n_과거회차 in range(n_기간):
            n_과거회차 = n_과거회차 + 1

            for s_번호 in self.li_45개번호:
                s_기존컬럼 = f'no{s_번호}'
                s_신규컬럼 = f'p{n_과거회차}_{s_기존컬럼}'
                df[s_신규컬럼] = df[s_기존컬럼].shift(n_과거회차)

        return df

    def _과거회차생성_2개조합(self, df, n_기간):
        """ 과거회차 당첨번호를 동일 열에 기입 """
        df = df.copy()

        # 과거회차별 컬럼 생성 (2개 번호 조합)
        for n_과거회차 in range(n_기간):
            n_과거회차 = n_과거회차 + 1

            for s_번호 in self.li_45개번호_2개조합:
                s_기존컬럼 = f'no{s_번호}'
                s_신규컬럼 = f'p{n_과거회차}_{s_기존컬럼}'
                df[s_신규컬럼] = df[s_기존컬럼].shift(n_과거회차)

        return df

    @staticmethod
    def _get_x(df, li_col_x):
        """ df 기준으로 x 정의 후 리턴 """
        df_x = df.loc[:, li_col_x].astype(int)
        ary_x = df_x.values

        return ary_x

    @staticmethod
    def _get_y(df, s_col_y):
        """ df 기준으로 y 정의 후 리턴 """
        df_y = df.loc[:, s_col_y].astype(int)
        ary_y = df_y.values

        return ary_y

    def _예측기준정보생성(self, s_기준일자):
        """ 예측 차수, 예측 추첨일 정보 생성 후 리턴 """
        # 이력 데이터 가져오기
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), engine='python').astype(str)
        df_이력 = df_이력.sort_values('date').reset_index(drop=True)

        # 예측용 데이터 준비
        s_기준일자 = pd.Timestamp(s_기준일자).strftime('%Y.%m.%d')
        df_데이터 = df_이력[df_이력['date'] <= s_기준일자]
        # df_데이터 = df_데이터[-1:]

        # 예측 기준정보 생성
        n_예측_차수 = int(df_데이터['seq'].values[-1]) + 1
        dt_예측_추첨일 = pd.Timestamp(df_데이터['date'].values[-1]) + pd.Timedelta(days=7)
        s_예측_추첨일 = dt_예측_추첨일.strftime('%Y%m%d')

        return n_예측_차수, s_예측_추첨일

    def _확률예측(self, ary_x, li_numbers, dic_pred_info):
        """ 저장된 model 을 불러와서 번호별 당첨 확률을 df 리턴 """
        dic_models = self.dic_args['dic_models']

        # 확률 예측 산출
        dic_확률 = {'no': [], 'prob_0': [], 'prob_1': []}
        for s_번호 in li_numbers:
            model = dic_models[f'model_no{s_번호}']
            n_확률값_0 = model.predict_proba(ary_x)[0][0]

            dic_확률['no'].append(f'no{s_번호}')
            dic_확률['prob_0'].append(n_확률값_0)
            dic_확률['prob_1'].append(1 - n_확률값_0)

        # 예측 정보 정리
        df_확률 = pd.DataFrame(dic_확률)
        df_확률['n_pred_seq'] = dic_pred_info['n_예측_차수']
        df_확률['s_pred_date'] = dic_pred_info['s_예측_추첨일']
        df_확률 = df_확률.sort_values('prob_1', ascending=False).reset_index(drop=True)

        return df_확률

    @staticmethod
    def _확률상위n개(df1, df2, n_top):
        """ 번호별 확률 df (1개짜리, 2개짜리) 가져와서 확률 기준 상위 5개에 해당하는 번호 추출하여 list 리턴 """
        # 상위 n개 번호 가져오기 (번호 1개)
        df_확률_1개_상위n개 = df1[:n_top].copy()
        df_확률_1개_상위n개['no_only'] = df_확률_1개_상위n개['no'].apply(lambda x: x.replace('no', ''))
        li_번호_상위n개_1개 = list(df_확률_1개_상위n개['no_only'].values)

        # 상위 n개 번호 가져오기 (번호 2개)
        df_확률_2개_상위n개 = df2[:n_top].copy()
        df_확률_2개_상위n개['no_only1'] = df_확률_2개_상위n개['no'].apply(lambda x: x[2:4])
        df_확률_2개_상위n개['no_only2'] = df_확률_2개_상위n개['no'].apply(lambda x: x[5:7])
        li_번호_상위n개_2개 = list(df_확률_2개_상위n개['no_only1'].values) + list(df_확률_2개_상위n개['no_only2'].values)

        # 상위 n개 번호 합치기 (번호 1개 + 번호 2개)
        sri_번호_상위n개 = pd.Series(li_번호_상위n개_1개 + li_번호_상위n개_2개)
        ary_번호_상위n개 = sri_번호_상위n개.sort_values().unique()
        li_번호_상위n개 = list(ary_번호_상위n개)
        return li_번호_상위n개

    @staticmethod
    def _번호조합(li_numbers):
        """ 입력받은 list 기준으로 6개 번호 조합을 경우의 수로 구하여 df 리턴 """
        li_번호조합 = []

        # 번호 조합
        for s_번호1 in li_numbers:
            li_numbers1 = [s for s in li_numbers if int(s) > int(s_번호1)]
            for s_번호2 in li_numbers1:
                li_numbers2 = [s for s in li_numbers1 if int(s) > int(s_번호2)]
                for s_번호3 in li_numbers2:
                    li_numbers3 = [s for s in li_numbers2 if int(s) > int(s_번호3)]
                    for s_번호4 in li_numbers3:
                        li_numbers4 = [s for s in li_numbers3 if int(s) > int(s_번호4)]
                        for s_번호5 in li_numbers4:
                            li_numbers5 = [s for s in li_numbers4 if int(s) > int(s_번호5)]
                            for s_번호6 in li_numbers5:
                                li_6개번호 = [s_번호1, s_번호2, s_번호3, s_번호4, s_번호5, s_번호6]
                                li_번호조합.append(li_6개번호)

        # df 정리
        li_컬럼명 = ['no1', 'no2', 'no3', 'no4', 'no5', 'no6']
        df_번호조합 = pd.DataFrame(li_번호조합, columns=li_컬럼명)

        return df_번호조합

    def _번호조합확률(self, df_numbers):
        """ 번호조합 df 입력받아 확률값 계산 후 df 리턴 """
        # 확률 분석 결과 가져오기
        df_확률_1개 = self.dic_args['df_확률_1개'].copy()
        df_확률_2개 = self.dic_args['df_확률_2개'].copy()
        dic_확률_1개 = df_확률_1개.set_index('no').to_dict()['prob_1']
        dic_확률_2개 = df_확률_2개.set_index('no').to_dict()['prob_1']
        dic_확률 = dict(dic_확률_1개, **dic_확률_2개)

        li_확률 = []
        # 번호 조합별 확률값 계산
        s_대상 = tqdm(df_numbers.values, desc='번호 조합별 확률 계산') if self.b_tqdm else df_numbers.values
        for ary_번호조합 in s_대상:
        # for ary_번호조합 in tqdm(df_numbers.values, desc='번호 조합별 확률 계산'):
            li_번호조합_n = [int(s) for s in ary_번호조합]
            # 2개씩 묶는 경우의 수 산출
            li_번호조합_2개_n = []
            for n_번호1 in li_번호조합_n:
                li_번호조합1_n = [n for n in li_번호조합_n if n > n_번호1]
                for n_번호2 in li_번호조합1_n:
                    li_2개번호_n = [n_번호1, n_번호2]
                    li_번호조합_2개_n.append(li_2개번호_n)

            # 경우의 수 산출
            li_번호조합_경우의수 = []

            # 번호 2개 묶음 0개일 때 경우의 수 산출 (1개)
            li_번호1개 = [f'no{s}' for s in ary_번호조합]
            li_번호조합_경우의수.append(li_번호1개)

            # 번호 2개 묶음 1개일 때 경우의 수 산출 (15개)
            for li_번호2개_n in li_번호조합_2개_n:
                li_번호1개_n = [n for n in li_번호조합_n if n not in li_번호2개_n]
                li_번호2개 = [f'no{n:02}' for n in li_번호1개_n] + [f'no{li_번호2개_n[0]:02}|{li_번호2개_n[1]:02}']
                li_번호조합_경우의수.append(li_번호2개)

            # 번호 2개 묶음 2개, 3개일 때 경우의 수 산출 (90개 + 90개)
            for li_번호2개_1_n in li_번호조합_2개_n:
                li_번호조합1_n = [n for n in li_번호조합_n if n not in li_번호2개_1_n]
                for n_번호1 in li_번호조합1_n:
                    li_번호조합2_n = [n for n in li_번호조합1_n if n > n_번호1]
                    for n_번호2 in li_번호조합2_n:
                        li_번호2개_2_n = [n_번호1, n_번호2]
                        li_번호1개_n = [n for n in li_번호조합_n if n not in li_번호2개_1_n]
                        li_번호1개_n = [n for n in li_번호1개_n if n not in li_번호2개_2_n]
                        li_번호2개_2개 = [f'no{n:02}' for n in li_번호1개_n]\
                                      + [f'no{li_번호2개_1_n[0]:02}|{li_번호2개_1_n[1]:02}']\
                                      + [f'no{li_번호2개_2_n[0]:02}|{li_번호2개_2_n[1]:02}']
                        li_번호조합_경우의수.append(li_번호2개_2개)

                        li_번호2개_3개 = [f'no{li_번호1개_n[0]:02}|{li_번호1개_n[1]:02}']\
                                      + [f'no{li_번호2개_1_n[0]:02}|{li_번호2개_1_n[1]:02}']\
                                      + [f'no{li_번호2개_2_n[0]:02}|{li_번호2개_2_n[1]:02}']
                        li_번호조합_경우의수.append(li_번호2개_3개)

            # 전체 경우의 수에 대한 확률 산출 후 max 값 선택
            li_확률_196개 = []
            for li_번호조합_s in li_번호조합_경우의수:
                li_확률_1개 = [dic_확률[s_번호] for s_번호 in li_번호조합_s]
                n_확률_6개번호 = np.prod(li_확률_1개)
                li_확률_196개.append(n_확률_6개번호)
            li_확률.append(np.max(li_확률_196개))

        # df_numbers 에 확률 추가
        df_numbers['prob_1'] = li_확률
        df_numbers = df_numbers.sort_values('prob_1', ascending=False).reset_index(drop=True)

        return df_numbers

    # @staticmethod
    # def _5세트선택_2개번호중최빈수(dic_확률):
    #     """ 계산된 확률값을 바탕으로 6개 번호 5 세트 선택 - 2개 번호 상위 30개 중 최빈수 선택 """
    #     # function 이름 찾기
    #     func_name = sys._getframe(0).f_code.co_name
    #     s_전략명 = func_name[1:]
    #
    #     # 확률값 가져오기
    #     df_확률_1개 = dic_확률['df_확률_1개']
    #     df_확률_2개 = dic_확률['df_확률_2개']
    #     df_확률_6개 = dic_확률['df_확률_6개']
    #
    #     # 2개 확률 중 상위 30개 번호 최빈수 찾기
    #     df_30개 = df_확률_2개[:30].copy()
    #     df_30개['no1'] = df_30개['no'].apply(lambda x: x[2:4])
    #     df_30개['no2'] = df_30개['no'].apply(lambda x: x[5:7])
    #     li_30개 = list(df_30개['no1'].values) + list(df_30개['no2'].values)
    #     sri_카운트 = pd.Series(li_30개).value_counts()
    #     s_최빈번호 = sri_카운트.index[0]
    #
    #     # 2개 번호 골라내기
    #     ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
    #     li_2개번호_전체 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if s_최빈번호 in ary_2개번호]
    #
    #     # 6개 번호조합에서 2개 번호 포함된 항목 골라내기
    #     ary_6개번호_전체 = df_확률_6개.loc[:, 'no1': 'no6'].values
    #     li_6개번호_전체 = [list(ary_6개번호) for ary_6개번호 in ary_6개번호_전체 if s_최빈번호 in ary_6개번호]
    #     li_6개번호_5개세트 = []
    #     for li_6개번호 in li_6개번호_전체:
    #         # 2개 번호 포함 여부 확인
    #         for li_2개번호 in li_2개번호_전체:
    #             if (li_2개번호[0] in li_6개번호) and (li_2개번호[1] in li_6개번호):
    #                 li_6개번호_5개세트.append(li_6개번호)
    #                 li_2개번호_전체.remove(li_2개번호)
    #                 break
    #
    #         if len(li_6개번호_5개세트) >= 5:
    #             break
    #
    #     # df 변환
    #     li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    #     df_6개번호_5개세트 = pd.DataFrame(li_6개번호_5개세트, columns=li_컬럼명)
    #
    #     # 확률값 추가
    #     df_확률_6개['no_all'] = ['_'.join(s_번호 for s_번호 in ary_6개번호) for ary_6개번호 in ary_6개번호_전체]
    #     dic_확률_6개 = df_확률_6개.set_index('no_all').to_dict()['prob_1']
    #     df_6개번호_5개세트['prob_1'] = [dic_확률_6개['_'.join(s_번호 for s_번호 in li_6개번호)] for li_6개번호 in li_6개번호_5개세트]
    #
    #     return s_전략명, df_6개번호_5개세트

    # @staticmethod
    # def _5세트선택_2개번호중최빈수_중복제외(dic_확률):
    #     """ 계산된 확률값을 바탕으로 6개 번호 5 세트 선택 - 2개 번호 상위 30개 중 최빈수 선택 """
    #     # function 이름 찾기
    #     func_name = sys._getframe(0).f_code.co_name
    #     s_전략명 = func_name[1:]
    #
    #     # 확률값 가져오기
    #     df_확률_1개 = dic_확률['df_확률_1개']
    #     df_확률_2개 = dic_확률['df_확률_2개']
    #     df_확률_6개 = dic_확률['df_확률_6개']
    #
    #     # 2개 확률 중 상위 30개 번호 최빈수 찾기
    #     df_30개 = df_확률_2개[:30].copy()
    #     df_30개['no1'] = df_30개['no'].apply(lambda x: x[2:4])
    #     df_30개['no2'] = df_30개['no'].apply(lambda x: x[5:7])
    #     li_30개 = list(df_30개['no1'].values) + list(df_30개['no2'].values)
    #     sri_카운트 = pd.Series(li_30개).value_counts()
    #     s_최빈번호 = sri_카운트.index[0]
    #
    #     # 2개 번호 골라내기
    #     ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
    #     li_2개번호_전체 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if s_최빈번호 in ary_2개번호]
    #     li_2번째번호 = []
    #     for li_2개번호 in li_2개번호_전체:
    #         for s_번호 in li_2개번호:
    #             if int(s_번호) is not int(s_최빈번호):
    #                 li_2번째번호.append(s_번호)
    #
    #     # 6개 번호조합에서 2개 번호 포함된 항목 골라내기
    #     ary_6개번호_전체 = df_확률_6개.loc[:, 'no1': 'no6'].values
    #     li_6개번호_전체 = [list(ary_6개번호) for ary_6개번호 in ary_6개번호_전체 if s_최빈번호 in ary_6개번호]
    #     li_6개번호_5개세트 = []
    #     li_2번째번호_지운거 = []
    #     for li_6개번호 in li_6개번호_전체:
    #         # 2번째 번호 포함 여부 확인
    #         for s_2번째번호 in li_2번째번호:
    #             if s_2번째번호 in li_6개번호:
    #                 # 사용한 번호 포함 여부 확인
    #                 li_사용한번호확인 = [1 if s_번호 in li_2번째번호_지운거 else 0 for s_번호 in li_6개번호]
    #                 if sum(li_사용한번호확인) > 0:
    #                     break
    #                 # 번호 추출
    #                 li_6개번호_5개세트.append(li_6개번호)
    #                 li_2번째번호.remove(s_2번째번호)
    #                 li_2번째번호_지운거.append(s_2번째번호)
    #                 break
    #
    #         if len(li_6개번호_5개세트) >= 5:
    #             break
    #
    #     # df 변환
    #     li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    #     df_6개번호_5개세트 = pd.DataFrame(li_6개번호_5개세트, columns=li_컬럼명)
    #
    #     # 확률값 추가
    #     df_확률_6개['no_all'] = ['_'.join(s_번호 for s_번호 in ary_6개번호) for ary_6개번호 in ary_6개번호_전체]
    #     dic_확률_6개 = df_확률_6개.set_index('no_all').to_dict()['prob_1']
    #     df_6개번호_5개세트['prob_1'] = [dic_확률_6개['_'.join(s_번호 for s_번호 in li_6개번호)] for li_6개번호 in li_6개번호_5개세트]
    #
    #     return s_전략명, df_6개번호_5개세트


######################################################################################################################
def 결과정리():
    """ result 폴더에 나온 결과를 합쳐서 하나의 csv 파일로 저장 """
    # 폴더 찾기
    folder_result = os.path.join(os.getcwd(), 'result')
    li_폴더 = [s_파일 for s_파일 in os.listdir(folder_result) if '.csv' not in s_파일]

    # 폴더별 결과 정리
    for s_폴더 in li_폴더:
        n_과거 = int(s_폴더[2:4])
        n_트리 = int(s_폴더[7:10])
        s_기준 = s_폴더

        # 파일 정보 읽어오기
        li_파일 = [s_파일 for s_파일 in os.listdir(os.path.join(folder_result, s_폴더))
                 if '_결과_' in s_파일 and '.csv' in s_파일]
        dic_결과 = {'차수': [], '추첨일': [], '일치갯수': [], '상금': []}
        for s_파일 in li_파일:
            li_정보 = s_파일.replace('.csv', '').split('_')
            dic_결과['차수'].append(str(li_정보[2].replace('차', '')))
            dic_결과['추첨일'].append(str(li_정보[3].replace('추첨', '')))
            dic_결과['일치갯수'].append(int(li_정보[4].replace('개', '')))
            dic_결과['상금'].append(int(li_정보[5].replace('원', '')))

        # df 정리
        df_결과 = pd.DataFrame(dic_결과)
        li_컬럼명 = list(df_결과.columns)
        df_결과['기준'] = s_기준
        li_컬럼명 = ['기준'] + li_컬럼명
        df_결과 = df_결과.loc[:, li_컬럼명].sort_values('추첨일', ascending=False)

        # csv 저장
        n_주 = len(df_결과)
        n_갯수 = df_결과['일치갯수'].max()
        n_상금 = df_결과['상금'].sum()
        s_파일명 = f'결과정리_{s_기준}_{n_주}주_{n_갯수}개_{n_상금}원.csv'
        df_결과.to_csv(os.path.join(folder_result, s_파일명), index=False, encoding='utf-8-sig')


def back_testing(s_date):
    """ 백테스팅용 실행 함수 (추첨이력_업데이트, 전처리_데이터변환 사전 실행 필요) """
    l = LottoAnal(s_date=s_date, b_tqdm=False)

    l.전처리_데이터변환()
    l.분석모델_생성()
    l.확률계산_1개2개조합()
    l.확률계산_6개조합()
    # l.번호선정_5세트()


######################################################################################################################
if __name__ == '__main__':
    # 실행용 코드
    s_날짜_오늘 = pd.Timestamp('now').strftime('%Y%m%d')
    l = LottoAnal(b_test=False, s_date=s_날짜_오늘)

    l.추첨이력_업데이트()
    l.전처리_데이터변환()
    l.분석모델_생성()
    l.확률계산_1개2개조합()
    l.확률계산_6개조합()
    l.번호선정_5세트(s_선정로직='번호선정_1개2개연계_최빈수연계')
    l.번호선정_5세트(s_선정로직='번호선정_2개번호_최빈수')

    # l.결과확인()


    # # 백테스팅 (확률값 생성, 시작하는 날짜의 다음 추첨일부터 생성)
    # n_주_테스트할거 = 883
    # li_날짜 = [pd.Timestamp('20190824') - pd.Timedelta(days=7 * n) for n in range(n_주_테스트할거)]
    # li_날짜 = [dt.strftime('%Y%m%d') for dt in li_날짜]
    # for s_날짜 in tqdm(li_날짜, desc=f'백테스팅_{n_주_테스트할거}주치'):
    #     if s_날짜 == '20171223':
    #         break
    #     back_testing(s_date=s_날짜)

############## 그 후에 기존 회차 확률 csv 파일들 활용하여 번호 선정 후 검증
