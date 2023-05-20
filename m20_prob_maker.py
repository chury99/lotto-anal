import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# noinspection PyUnresolvedReferences,PyPep8Naming,PyProtectedMember
class ProbMaker:
    def __init__(self, s_기준일=None, b_test=False):
        # config 읽어 오기
        with open('config.json', mode='rt', encoding='utf-8') as file:
            dic_config = json.load(file)

        # 기준 정보 설정
        self.s_오늘 = pd.Timestamp('now').strftime('%Y%m%d')
        self.path_log = os.path.join(dic_config['folder_log'], f'log_lotto_{self.s_오늘}.log')
        self.s_기준일 = self.s_오늘 if s_기준일 is None else s_기준일
        self.li_45개번호 = [f'{i + 1:02}' for i in range(45)]
        self.li_45개번호_2개조합 = self._번호조합_2개(li_전체번호=self.li_45개번호)
        self.li_6개순번_2개조합 = self._순번조합_2개()
        self.n_과거회차 = int(dic_config['고려할과거회차'])
        self.n_학습차수 = int(dic_config['학습진행차수'])
        self.b_test = b_test

        # 폴더 설정
        folder_work = dic_config['folder_work']
        self.folder_history = os.path.join(folder_work, 'history')
        self.folder_run = os.path.join(folder_work, 'run')
        self.folder_result = os.path.join(folder_work, 'result')
        self.folder_확률예측 = os.path.join(self.folder_run, '확률예측')
        os.makedirs(self.folder_history, exist_ok=True)
        os.makedirs(self.folder_run, exist_ok=True)
        os.makedirs(self.folder_result, exist_ok=True)
        os.makedirs(self.folder_확률예측, exist_ok=True)

        # dic_args 설정
        self.path_args = os.path.join(self.folder_run, 'dic_args.pkl')
        if 'dic_args.pkl' in os.listdir(self.folder_run):
            self.dic_args = pd.read_pickle(self.path_args)
        else:
            self.dic_args = dict()

        # log 기록
        self.make_log(f'### 확률 계산 시작 (기준일-{self.s_기준일}) ###')

    def 전처리_데이터변환(self):
        """ 추첨이력 데이터 변환 (one-hot encoding, 2개 조합, 과거 데이터 생성) """
        # 로그 기록
        self.make_log('# 데이터 전처리 #')

        # 데이터 불러오기
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), encoding='cp949').astype(str)

        # 데이터 정렬 (일자순)
        df_이력 = df_이력.sort_values('추첨일').reset_index(drop=True)

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
        if self.b_test:
            pd.to_pickle(self.dic_args, self.path_args)

    def 분석모델_생성(self):
        """ x, y 데이터셋 생성 및 Random Forest 학습 모델 생성 """
        # 로그 기록
        self.make_log(f'# 분석모델 생성_{self.s_기준일} #')

        # 전처리 데이터 불러오기
        df_이력 = self.dic_args['df_이력']

        # 검증구간 데이터 잘라내기
        s_기준일 = pd.Timestamp(self.s_기준일).strftime('%Y.%m.%d')
        df_데이터 = df_이력[df_이력['추첨일'] <= s_기준일].copy()
        df_데이터 = df_데이터[(-1 * self.n_학습차수):]

        # x 데이터셋 생성
        li_컬럼명_x = [s for s in df_데이터.columns if s.startswith('p')]
        ary_x_학습 = self._get_x(df=df_데이터, li_col_x=li_컬럼명_x)

        # 모델 저장용 dic 생성
        dic_models = dict()

        # 45개 번호별 모델 생성
        s_대상 = tqdm(self.li_45개번호, desc='45개 번호별 학습') if self.b_test else self.li_45개번호
        for s_번호 in s_대상:
            ary_y_학습 = self._get_y(df=df_데이터, s_col_y=f'no{s_번호}')

            # random forest 모델 학습
            model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=99)
            model.fit(ary_x_학습, ary_y_학습)

            # 모델 저장
            dic_models[f'model_no{s_번호}'] = model

        # 2개 조합 번호별 모델 생성
        s_대상 = tqdm(self.li_45개번호_2개조합, desc='2개 조합별 학습') if self.b_test else self.li_45개번호_2개조합
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
        if self.b_test:
            pd.to_pickle(self.dic_args, self.path_args)

    def 확률계산_1개2개조합(self):
        """ 분석모델을 통해 번호별 확률 계산 후 result 폴더에 csv 저장 """
        # 로그 기록
        self.make_log(f'# 번호별 확률 계산 (1개, 2개 조합)_{self.s_기준일} #')

        # 예측용 데이터 준비
        df_이력 = self.dic_args['df_이력']
        s_기준일 = pd.Timestamp(self.s_기준일).strftime('%Y.%m.%d')
        df_데이터 = df_이력[df_이력['추첨일'] <= s_기준일]
        df_데이터 = df_데이터[-1:]

        # 예측 기준정보 생성
        n_예측_차수, s_예측_추첨일 = self._예측기준정보생성(s_기준일=self.s_기준일)
        dic_예측_기준정보 = dict()
        dic_예측_기준정보['n_예측_차수'] = n_예측_차수
        dic_예측_기준정보['s_예측_추첨일'] = s_예측_추첨일

        # 예측용 x 데이터셋 생성
        li_컬럼명_제외 = (['회차', '추첨일']
                     + [s for s in df_데이터.columns if s.startswith('win')]
                     + [s for s in df_데이터.columns if s.startswith(f'p{self.n_과거회차}')])
        li_컬럼명_x = [s for s in df_데이터.columns if s not in li_컬럼명_제외]
        ary_x_예측 = df_데이터.loc[:, li_컬럼명_x].values

        # 45개 번호 확률 예측
        df_확률_1개 = self._확률예측(ary_x=ary_x_예측, li_numbers=self.li_45개번호, dic_pred_info=dic_예측_기준정보)
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_1개.to_csv(os.path.join(self.folder_확률예측, s_파일명), index=False)

        # 2개 조합 번호 확률 예측
        df_확률_2개 = self._확률예측(ary_x=ary_x_예측, li_numbers=self.li_45개번호_2개조합, dic_pred_info=dic_예측_기준정보)
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_2개.to_csv(os.path.join(self.folder_확률예측, s_파일명), index=False)

        # 결과 저장
        self.dic_args['df_확률_1개'] = df_확률_1개
        self.dic_args['df_확률_2개'] = df_확률_2개
        self.dic_args['n_예측_차수'] = n_예측_차수
        self.dic_args['s_예측_추첨일'] = s_예측_추첨일

        # dic_args 저장
        if self.b_test:
            pd.to_pickle(self.dic_args, self.path_args)

    def 확률계산_6개조합(self):
        """ 1개, 2개 확률 상위 n개씩 번호 조합하여 6개 번호의 확률 산출 후 result 폴더에 csv 저장 """
        # 예측 기준정보 생성
        n_예측_차수, s_예측_추첨일 = self._예측기준정보생성(s_기준일=self.s_기준일)

        # 로그 기록
        self.make_log(f'# 번호 6개 조합별 확률 산출 ({n_예측_차수:,}차_{s_예측_추첨일}추첨) #')

        # 확률 분석 결과 가져오기
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_1개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        df_확률_2개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
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
        df_번호조합확률.to_csv(os.path.join(self.folder_확률예측, s_파일명), index=False)

        # 결과 저장
        self.dic_args['df_번호조합확률'] = df_번호조합확률

        # dic_args 저장
        if self.b_test:
            pd.to_pickle(self.dic_args, self.path_args)

    ###################################################################################################################

    def make_log(self, s_text, li_loc=None):
        """ 입력 받은 s_text 에 시간 붙여서 self.path_log 에 저장 """
        # 정보 설정
        s_시각 = pd.Timestamp('now').strftime('%H:%M:%S')
        s_파일 = os.path.basename(sys.argv[0]).replace('.py', '')
        s_모듈 = sys._getframe(1).f_code.co_name

        # log 생성
        s_log = f'[{s_시각}] {s_파일} | {s_모듈} | {s_text}'

        # log 출력
        li_출력 = ['콘솔', '파일'] if li_loc is None else li_loc
        if '콘솔' in li_출력:
            print(s_log)
        if '파일' in li_출력:
            with open(self.path_log, mode='at', encoding='cp949') as file:
                file.write(f'{s_log}\n')

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
            df[s_순번조합] = df[s_순번1].apply(lambda x: f'{int(x):02}') + '|' + df[s_순번2].apply(
                lambda x: f'{int(x):02}')

        return df

    def _ohe_당첨번호(self, df):
        """ 당첨번호 6개를 one-hot encoding 으로 변환 """
        df = df.copy()

        # 45개 번호에 대해 one-hot encoding 진행
        for s_번호_확인할거 in self.li_45개번호:
            df[f'no{s_번호_확인할거}'] = 0

            # 6개 당첨번호 에 대해 해당 번호에 1 표기
            for n_순번 in range(1, 7):
                df.loc[df[f'win_{n_순번}'] == s_번호_확인할거, f'no{s_번호_확인할거}'] = 1

        return df

    def _과거회차생성(self, df, n_기간):
        """ 과거회차 당첨번호 를 동일 열에 기입 """
        df = df.copy()

        # 과거 회차별 컬럼 생성 (1개 번호) (빠른 버전)
        df_new = pd.concat(
                            (
                                pd.Series(
                                            (
                                                df[f'no{s_번호}'].shift(n_과거회차)
                                            ), name=f'p{n_과거회차}_no{s_번호}'
                                         ) for n_과거회차 in range(1, n_기간 + 1) for s_번호 in self.li_45개번호
                            ), axis=1
                          )
        df = pd.concat([df, df_new], axis=1)

        # 과거 회차별 컬럼 생성 (1개 번호) (쉬운 버전)
        # for n_과거회차 in range(n_기간):
        #     n_과거회차 = n_과거회차 + 1
        #
        #     for s_번호 in self.li_45개번호:
        #         s_기존컬럼 = f'no{s_번호}'
        #         s_신규컬럼 = f'p{n_과거회차}_{s_기존컬럼}'
        #         df[s_신규컬럼] = df[s_기존컬럼].shift(n_과거회차)

        return df

    def _ohe_당첨번호_2개조합(self, df):
        """ 당첨번호 6개를 2개씩 조합하여 one-hot encoding 으로 변환 """
        df = df.copy()

        # 45개 번호에 대한 2개 조합 번호에 대해 one-hot encoding 진행 (빠른 버전)
        ary_당첨번호_전체 = df.loc[:, 'win_1':'win_6'].astype(int).values
        df_new = pd.concat(
                            (pd.Series(
                                         (
                                           (int(s_번호_확인할거[:2]) in ary and int(s_번호_확인할거[3:5]) in ary) * 1
                                           for ary in ary_당첨번호_전체
                                         ), name=f'no{s_번호_확인할거}'
                                      ) for s_번호_확인할거 in self.li_45개번호_2개조합
                             ), axis=1
                          )
        # 기존 df와 합치기
        df = pd.concat([df, df_new], axis=1)

        # 45개 번호에 대한 2개 조합 번호에 대해 one-hot encoding 진행 (쉬운 버전)
        # for s_번호_확인할거 in self.li_45개번호_2개조합:
        #     df[f'no{s_번호_확인할거}'] = 0
        #
        #     # 순번 2개 조합(15개)에 대해 해당 번호에 1 표기
        #     for s_순번조합 in self.li_6개순번_2개조합:
        #         df.loc[df[f'win_{s_순번조합}'] == s_번호_확인할거, f'no{s_번호_확인할거}'] = 1

        # 45개 번호에 대한 2개 조합 번호에 대해 one-hot encoding 진행 (중간 버전)
        # ary_당첨번호_전체 = df.loc[:, 'win_1':'win_6'].astype(int).values
        # li_컬럼명 = []
        # li_컬럼값 = []
        # for s_번호_확인할거 in self.li_45개번호_2개조합:
        #     n_번호1 = int(s_번호_확인할거[:2])
        #     n_번호2 = int(s_번호_확인할거[3:5])
        #     # df2[f'no{s_번호_확인할거}'] = [((n_번호1 in ary) and (n_번호2 in ary)) * 1 for ary in ary_당첨번호_전체]
        #     li_컬럼명.append(f'no{s_번호_확인할거}')
        #     li_컬럼값.append(((n_번호1 in ary) and (n_번호2 in ary)) * 1 for ary in ary_당첨번호_전체)

        return df

    def _과거회차생성_2개조합(self, df, n_기간):
        """ 과거회차 당첨번호 를 동일 열에 기입 """
        df = df.copy()

        # 과거 회차별 컬럼 생성 (2개 번호 조합) (빠른 버전)
        df_new = pd.concat(
                            (
                                pd.Series(
                                            (
                                                df[f'no{s_번호}'].shift(n_과거회차)
                                            ), name=f'p{n_과거회차}_no{s_번호}'
                                ) for n_과거회차 in range(1, n_기간 + 1) for s_번호 in self.li_45개번호_2개조합
                            ), axis=1
                          )
        df = pd.concat([df, df_new], axis=1)

        # 과거 회차별 컬럼 생성 (2개 번호 조합) (쉬운 버전)
        # for n_과거회차 in range(n_기간):
        #     n_과거회차 = n_과거회차 + 1
        #
        #     for s_번호 in self.li_45개번호_2개조합:
        #         s_기존컬럼 = f'no{s_번호}'
        #         s_신규컬럼 = f'p{n_과거회차}_{s_기존컬럼}'
        #         df[s_신규컬럼] = df[s_기존컬럼].shift(n_과거회차)

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

    def _예측기준정보생성(self, s_기준일):
        """ 예측 차수, 예측 추첨일 정보 생성 후 리턴 """
        # 이력 데이터 가져오기
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), encoding='cp949').astype(str)
        df_이력 = df_이력.sort_values('추첨일').reset_index(drop=True)

        # 예측용 데이터 준비
        s_기준일 = pd.Timestamp(s_기준일).strftime('%Y.%m.%d')
        df_데이터 = df_이력[df_이력['추첨일'] <= s_기준일]
        # df_데이터 = df_데이터[-1:]

        # 예측 기준정보 생성
        n_예측_차수 = int(df_데이터['회차'].values[-1]) + 1
        dt_예측_추첨일 = pd.Timestamp(df_데이터['추첨일'].values[-1]) + pd.Timedelta(days=7)
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
        """ 입력 받은 list 기준으로 6개 번호 조합을 경우의 수로 구하여 df 리턴 """
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
        """ 번호조합 df 입력 받아 확률값 계산 후 df 리턴 """
        # 확률 분석 결과 가져오기
        df_확률_1개 = self.dic_args['df_확률_1개'].copy()
        df_확률_2개 = self.dic_args['df_확률_2개'].copy()
        dic_확률_1개 = df_확률_1개.set_index('no').to_dict()['prob_1']
        dic_확률_2개 = df_확률_2개.set_index('no').to_dict()['prob_1']
        dic_확률 = dict(dic_확률_1개, **dic_확률_2개)

        li_확률 = []
        # 번호 조합별 확률값 계산
        s_대상 = tqdm(df_numbers.values, desc='번호 조합별 확률 계산') if self.b_test else df_numbers.values
        for ary_번호조합 in s_대상:
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


#######################################################################################################################
if __name__ == '__main__':
    # 기준일 없으면 오늘 날짜, b_test 없으면 False
    # p = ProbMaker(s_기준일='20230331', b_test=True)
    p = ProbMaker(s_기준일=None)
    p.전처리_데이터변환()
    p.분석모델_생성()
    p.확률계산_1개2개조합()
    p.확률계산_6개조합()
