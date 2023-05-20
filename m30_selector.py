import os
import sys
import json
import pandas as pd

import m31_selector_algorithm as logic


# noinspection PyUnresolvedReferences,PyPep8Naming,PyProtectedMember
class Selector:
    def __init__(self):
        # config 읽어 오기
        with open('config.json', mode='rt', encoding='utf-8') as file:
            dic_config = json.load(file)

        # 기준 정보 설정
        self.s_오늘 = pd.Timestamp('now').strftime('%Y%m%d')
        self.path_log = os.path.join(dic_config['folder_log'], f'log_lotto_{self.s_오늘}.log')

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

        # log 기록
        self.make_log(f'### 번호 선정 시작 ###')

        # 회차별 추첨일 데이터 생성
        self.dic_추첨일 = self._회차별추첨일생성()

        # 대상 회차 생성
        self.li_회차_대상 = self._대상회차생성()

    def 번호선정(self, s_선정로직):
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더_선정결과 = os.path.join(self.folder_번호선정, f'번호선정로직_{s_선정로직[5:]}')
        os.makedirs(s_폴더_선정결과, exist_ok=True)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더_선정결과)
                  if '확률예측_번호선정_' in s_파일명 and '.csv' in s_파일명 and 'summary_' not in s_파일명]
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_회차_완료 = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]

        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_대상 if n_회차 not in li_회차_완료]

        for n_회차 in li_회차_할거:
            s_추첨일 = self.dic_추첨일[n_회차]

            # 로그 기록
            self.make_log(f'# {s_선정로직} ({n_회차:,}차_{s_추첨일}추첨) #')

            # 전달할 정보 설정
            dic_정보 = self._확률값_읽어오기(n_예측_차수=n_회차, s_예측_추첨일=s_추첨일)
            dic_정보['n_회차'] = n_회차
            dic_정보['s_추첨일'] = s_추첨일
            dic_정보['folder_번호선정'] = self.folder_번호선정

            # 번호 선정
            s_전략명 = None
            df_6개번호_5개세트 = None
            if s_선정로직 == '번호선정_2개번호_최빈수':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_최빈수(dic_정보=dic_정보)
            if s_선정로직 == '번호선정_2개번호_최빈수_중복제외':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_최빈수_중복제외(dic_정보=dic_정보)
            if s_선정로직 == '번호선정_2개번호_따라가기':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_따라가기(dic_정보=dic_정보)
            if s_선정로직 == '번호선정_2개번호_따라가기_확률반영':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_2개번호_따라가기_확률반영(dic_정보=dic_정보)
            if s_선정로직 == '번호선정_복합로직_최빈수_따라가기':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_복합로직_최빈수_따라가기(dic_정보=dic_정보)
            if s_선정로직 == '번호선정_1개2개연계_최빈수연계':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_1개2개연계_최빈수연계(dic_정보=dic_정보)
            if s_선정로직 == '번호선정_앙상블_결과종합':
                s_전략명, df_6개번호_5개세트 = logic.번호선정로직_앙상블_결과종합(dic_정보=dic_정보)

            # 결과 저장용 폴더 생성
            folder_전략명 = os.path.join(self.folder_번호선정, s_전략명)
            os.makedirs(folder_전략명, exist_ok=True)

            # csv 저장
            s_파일명 = f'확률예측_번호선정_{n_회차}차_{s_추첨일}추첨.csv'
            df_6개번호_5개세트.to_csv(os.path.join(folder_전략명, s_파일명), index=False, encoding='cp949')

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

    def _확률값_읽어오기(self, n_예측_차수, s_예측_추첨일):
        """ 최종 구매할 6개 번호 * 5개 세트 선정 후 result 폴더에 csv 저장 (기존 데이터는 csv로 읽어 오기) """
        # 확률 분석 결과 가져오기
        dic_확률 = dict()
        s_파일명 = f'확률예측_1개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_1개'] = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        s_파일명 = f'확률예측_2개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_2개'] = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
        s_파일명 = f'확률예측_6개번호_{n_예측_차수}차_{s_예측_추첨일}추첨.csv'
        dic_확률['df_확률_6개'] = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')

        return dic_확률

    def _회차별추첨일생성(self):
        """ 이력 데이터 불러와서 다음 회차 추가한 후 회차별 추첨일 데이터 생성 후 dict 리턴 """
        # 이력 데이터 불러오기
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), encoding='cp949')
        df_이력['추첨일'] = df_이력['추첨일'].apply(lambda x: x.replace('.', ''))

        # dic 생성
        dic_추첨일 = df_이력.set_index('회차').to_dict()['추첨일']

        # 다음 회차 데이터 생성
        n_회차_마지막 = df_이력['회차'].max()
        n_회차_다음 = n_회차_마지막 + 1
        dt_추첨일_다음 = pd.Timestamp(dic_추첨일[n_회차_마지막]) + pd.Timedelta(days=7)
        n_추첨일_다음 = dt_추첨일_다음.strftime('%Y%m%d')

        # 다음 회차 데이터 추가
        dic_추첨일[n_회차_다음] = n_추첨일_다음

        return dic_추첨일

    def _대상회차생성(self):
        """ 6개 확률 파일 확인해서 대상 회차 List 리턴 """
        # 6개 확률 파일명 불러오기
        s_폴더_확인대상 = os.path.join(self.folder_확률예측)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더_확인대상)
                  if '확률예측_6개번호_' in s_파일명 and '.csv' in s_파일명 and 'summary_' not in s_파일명]

        # 회차 정보만 골라내기
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_회차_대상 = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]

        return li_회차_대상


#######################################################################################################################
if __name__ == '__main__':
    # 6개 확률 데이터 기준 빈 날짜 자동 생성
    s = Selector()
    s.번호선정(s_선정로직='번호선정_2개번호_최빈수')
    s.번호선정(s_선정로직='번호선정_2개번호_최빈수_중복제외')
    s.번호선정(s_선정로직='번호선정_2개번호_따라가기')
    s.번호선정(s_선정로직='번호선정_2개번호_따라가기_확률반영')
    s.번호선정(s_선정로직='번호선정_복합로직_최빈수_따라가기')
    s.번호선정(s_선정로직='번호선정_1개2개연계_최빈수연계')
    s.번호선정(s_선정로직='번호선정_앙상블_결과종합')
