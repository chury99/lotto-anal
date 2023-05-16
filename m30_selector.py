import os
import sys
import json
import pandas as pd

import m31_selector_algorithm as logic


# noinspection PyUnresolvedReferences,PyPep8Naming,PyProtectedMember
class Selector:
    def __init__(self, s_기준일=None, b_test=False):
        # config 읽어 오기
        with open('config.json', mode='rt', encoding='utf-8') as file:
            dic_config = json.load(file)

        # 기준 정보 설정
        self.s_오늘 = pd.Timestamp('now').strftime('%Y%m%d')
        self.path_log = os.path.join(dic_config['folder_log'], f'log_lotto_{self.s_오늘}.log')
        self.s_기준일 = self.s_오늘 if s_기준일 is None else s_기준일
        self.b_test = b_test

        # 폴더 설정
        folder_work = dic_config['folder_work']
        self.folder_history = os.path.join(folder_work, 'history')
        self.folder_run = os.path.join(folder_work, 'run')
        self.folder_result = os.path.join(folder_work, 'result')
        self.folder_확률예측 = os.path.join(self.folder_run, '확률예측')
        self.folder_번호선정 = os.path.join(self.folder_result, '번호선정')
        os.makedirs(self.folder_history, exist_ok=True)
        os.makedirs(self.folder_run, exist_ok=True)
        os.makedirs(self.folder_result, exist_ok=True)
        os.makedirs(self.folder_확률예측, exist_ok=True)
        os.makedirs(self.folder_번호선정, exist_ok=True)

        # dic_args 설정
        self.path_args = os.path.join(self.folder_run, 'dic_args.pkl')
        if 'dic_args.pkl' in os.listdir(self.folder_run):
            self.dic_args = pd.read_pickle(self.path_args)
        else:
            self.dic_args = dict()

        # 카카오 API 연결
        # sys.path.extend(dic_config['folder_kakao'])
        # import API_kakao

        # log 기록
        self.make_log(f'### 번호 선정 시작 (기준일-{self.s_기준일}) ###')

        # 예측 기준정보 생성
        self.n_예측_차수, self.s_예측_추첨일 = self._예측기준정보생성(s_기준일=self.s_기준일)

        # 확률값 읽어오기
        self.dic_확률 = self._확률값_읽어오기()

    def 번호선정(self, s_선정로직):
        # 로그 기록
        self.make_log(f'# {s_선정로직} ({self.n_예측_차수:,}차_{self.s_예측_추첨일}추첨) #')

        # 번호 선정
        s_전략명 = None
        df_6개번호_5개세트 = None
        dic_확률 = self.dic_확률
        if s_선정로직 == '번호선정_2개번호_최빈수':
            s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_최빈수(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_2개번호_최빈수_중복제외':
            s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_최빈수_중복제외(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_2개번호_따라가기':
            s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_따라가기(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_2개번호_따라가기_확률반영':
            s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_따라가기_확률반영(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_복합로직_최빈수_따라가기':
            s_전략명, df_6개번호_5개세트 = logic.번호선정로직_복합로직_최빈수_따라가기(dic_확률=dic_확률)
        if s_선정로직 == '번호선정_1개2개연계_최빈수연계':
            s_전략명, df_6개번호_5개세트 = logic.번호선정로직_1개2개연계_최빈수연계(dic_확률=dic_확률)

        # 결과 저장용 폴더 생성
        folder_전략명 = os.path.join(self.folder_번호선정, s_전략명)
        os.makedirs(folder_전략명, exist_ok=True)

        # csv 저장
        s_파일명 = f'확률예측_번호선정_{self.n_예측_차수}차_{self.s_예측_추첨일}추첨.csv'
        df_6개번호_5개세트.to_csv(os.path.join(folder_전략명, s_파일명), index=False)

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

    def _확률값_읽어오기(self):
        """ 최종 구매할 6개 번호 * 5개 세트 선정 후 result 폴더에 csv 저장 (기존 데이터는 csv로 읽어 오기) """
        # 예측 기준정보 정의
        n_예측_차수 = self.n_예측_차수
        s_예측_추첨일 = self.s_예측_추첨일

        # 확률 분석 결과 가져오기
        dic_확률 = dict()
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_1개'] = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_2개'] = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        s_파일명 = f'확률예측_6개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_6개'] = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        # df_확률_6개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        # for n in range(6):
        #     df_확률_6개[f'no{n + 1}'] = df_확률_6개[f'no{n + 1}'].apply(lambda x: f'{x:02}')
        # dic_확률['df_확률_6개'] = df_확률_6개

        return dic_확률

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


#######################################################################################################################
if __name__ == '__main__':
    s = Selector(b_test=True)
    s.번호선정(s_선정로직='번호선정_2개번호_최빈수')
    s.번호선정(s_선정로직='번호선정_2개번호_최빈수_중복제외')
    s.번호선정(s_선정로직='번호선정_2개번호_따라가기')
    s.번호선정(s_선정로직='번호선정_2개번호_따라가기_확률반영')
    s.번호선정(s_선정로직='번호선정_복합로직_최빈수_따라가기')
    s.번호선정(s_선정로직='번호선정_1개2개연계_최빈수연계')
