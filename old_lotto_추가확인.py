import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 개인화 모듈 import
sys.path.extend(['D:\\_python@local\\ShortPunchTrader'])
import _custom_module_32 as cm
import API_kakao
import old_lotto_logic


# noinspection PyPep8Naming,PyShadowingNames
class LottoAppendix:
    def __init__(self):
        # 폴더 정의
        self.folder = os.getcwd()
        self.folder_history = os.path.join(self.folder, 'history')
        self.folder_run = os.path.join(self.folder, 'run')
        self.folder_result = os.path.join(self.folder, 'result')
        self.folder_result_확률예측 = os.path.join(self.folder_result, '확률예측')
        self.folder_result_추가확인 = os.path.join(self.folder_result, '추가확인')
        os.makedirs(self.folder_history, exist_ok=True)
        os.makedirs(self.folder_run, exist_ok=True)
        os.makedirs(self.folder_result, exist_ok=True)
        os.makedirs(self.folder_result_확률예측, exist_ok=True)
        os.makedirs(self.folder_result_추가확인, exist_ok=True)

        # path 정의
        self.path_args = os.path.join(self.folder_run, 'dic_args.pkl')

        # self 변수 정의 (당첨이력)
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), engine='python')
        self.df_이력 = df_이력.sort_values('seq', ascending=False)

        # self 변수 정의 (회차 list)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(self.folder_result_확률예측)
                  if '확률예측_2개번호_' in s_파일명 and '.csv' in s_파일명]
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        self.li_회차_n = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]
        self.li_회차_n.sort()

        # dic_args 정의
        if 'dic_args.pkl' in os.listdir(self.folder_run):
            self.dic_args = pd.read_pickle(self.path_args)
        else:
            self.dic_args = {}

    def 포함확인_당첨번호(self, n_번호확인갯수):
        """ 후보 숫자들이 당첨번호 안에 포함되는지 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_당첨번호')
        os.makedirs(s_폴더명, exist_ok=True)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                  if '당첨번호포함여부_' in s_파일명 and '.csv' in s_파일명 and 'summary' not in s_파일명]
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_회차_완료 = [int(s_정보[1].replace('차', '')) for s_정보 in li_정보]
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in li_회차_완료]

        # 회차별 포함 여부 확인
        for n_회차 in tqdm(li_회차_할거, desc='포함확인_당첨번호'):
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['seq'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['date'].values[0].replace('.', '')
            ary_당첨_n = df_이력_회차.loc[:, 'win_1': 'win_6'].values[0]

            # 예측번호 1개 확인
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측1개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            ary_예측번호_1개_s = df_예측1개[:n_번호확인갯수]['no'].values
            li_예측번호_1개_n = [int(s[2:]) for s in ary_예측번호_1개_s]

            # 예측번호 2개 확인
            s_파일명 = f'확률예측_2개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측2개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            ary_예측번호_2개_s = df_예측2개[:n_번호확인갯수]['no'].values
            li_예측번호_2개_n = [int(s[2:4]) for s in ary_예측번호_2개_s] + [int(s[5:]) for s in ary_예측번호_2개_s]

            # 예측번호 합치기 (중복 제거)
            li_예측번호_n = li_예측번호_1개_n + li_예측번호_2개_n
            li_예측번호_n = list(pd.Series(li_예측번호_n).unique())

            # df로 정리
            df_포함여부 = pd.DataFrame({'pred_no': li_예측번호_n})
            df_포함여부['win_ok'] = [1 if n in ary_당첨_n else 0 for n in li_예측번호_n]

            # 추가 정보 정리
            n_번호갯수 = len(df_포함여부)
            n_포함갯수 = df_포함여부['win_ok'].sum()

            # csv 저장
            s_파일명 = f'당첨번호포함여부_{n_회차}차_{s_추첨일}추첨_상위{n_번호확인갯수}개확인_{n_번호갯수}개추출_{n_포함갯수}개포함.csv'
            s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_당첨번호')
            df_포함여부.to_csv(os.path.join(s_폴더명, s_파일명), index=False)

        # 당첨번호 포함 확인 csv 파일 읽어오기
        s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_당첨번호')
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                  if '당첨번호포함여부_' in s_파일명 and '.csv' in s_파일명 and '_summary' not in s_파일명]

        # 정보 정리
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_정보 = [[int(s_정보[1].replace('차', '')),
                  s_정보[2].replace('추첨', ''),
                  s_정보[3].replace('상위', '').replace('개확인', ''),
                  s_정보[4].replace('개추출', ''),
                  s_정보[5].replace('개포함.csv', '')]
                 for s_정보 in li_정보]

        # df 생성
        li_컬럼명 = ['seq', 'date', 'top_cnt', 'no_cnt', 'win_cnt']
        df_정리 = pd.DataFrame(li_정보, columns=li_컬럼명)
        df_정리 = df_정리.sort_values('seq', ascending=False).reset_index(drop=True)

        # csv 저장
        s_파일명 = '당첨번호포함여부_summary.csv'
        df_정리.to_csv(os.path.join(s_폴더명, s_파일명), index=False)

    def 포함확인_확률번호(self):
        """ 확률 1개, 2개 번호 중 상위 6개 숫자가 당첨번호에 포함되는지 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_확률번호')
        os.makedirs(s_폴더명, exist_ok=True)
        s_파일명 = '확률번호포함확인_summary.csv'
        try:
            df_summary_기존 = pd.read_csv(os.path.join(s_폴더명, s_파일명), engine='python')
        except FileNotFoundError:
            df_summary_기존 = pd.DataFrame({'seq': [self.li_회차_n[0]]})
        ary_회차_완료 = df_summary_기존['seq'].values
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in ary_회차_완료]

        # 회차별 포함 여부 확인
        li_포함여부 = []
        for n_회차 in tqdm(li_회차_할거, desc='포함확인_확률번호'):
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['seq'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['date'].values[0].replace('.', '')
            li_당첨_n = list(df_이력_회차.loc[:, 'seq': 'win_6'].values[0])

            # 당첨번호포함여부 확인
            s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_당첨번호')
            li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                      if f'당첨번호포함여부_{n_회차}차_' in s_파일명 and '.csv' in s_파일명]
            n_당첨번호포함갯수 = int(li_파일명[0].split('_')[5].replace('개포함.csv', ''))

            # 예측번호 1개 확인
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측1개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')

            li_예측번호_1개 = []
            li_포함여부_1개 = []
            for i in range(6):
                n_예측1개 = int(df_예측1개['no'].values[i].replace('no', ''))
                li_예측번호_1개.append(n_예측1개)
                if n_예측1개 in li_당첨_n:
                    li_포함여부_1개.append(1)
                else:
                    li_포함여부_1개.append(0)

            n_포함여부_1개_합계 = sum(li_포함여부_1개)

            # 예측번호 2개 확인
            s_파일명 = f'확률예측_2개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측2개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')

            li_예측번호_2개 = []
            li_포함여부_2개 = []
            for i in range(6):
                s_예측2개 = df_예측2개['no'].values[i].replace('no', '')
                n_예측2개_1 = int(s_예측2개.split('|')[0])
                n_예측2개_2 = int(s_예측2개.split('|')[1])
                li_예측번호_2개.append(s_예측2개)
                if (n_예측2개_1 in li_당첨_n) and (n_예측2개_2 in li_당첨_n):
                    li_포함여부_2개.append(1)
                else:
                    li_포함여부_2개.append(0)

            n_포함여부_2개_합계 = sum(li_포함여부_2개)

            # 결과 데이터 생성
            li_결과 = li_당첨_n + [n_당첨번호포함갯수]\
                    + li_예측번호_1개 + li_포함여부_1개 + [n_포함여부_1개_합계]\
                    + li_예측번호_2개 + li_포함여부_2개 + [n_포함여부_2개_합계]

            # 추첨번호에 포함 여부 확인
            li_포함여부.append(li_결과)

        # 포함여부 확인 결과 df 정리
        li_컬럼명 = ['seq', 'date', 'win1', 'win2', 'win3', 'win4', 'win5', 'win6'] + ['cnt_in']\
                 + [f'1ea_no_{n + 1}' for n in range(6)] + [f'1ea_in_{n+1}' for n in range(6)] + ['cnt_in_1ea']\
                 + [f'2ea_no_{n + 1}' for n in range(6)] + [f'2ea_in_{n+1}' for n in range(6)] + ['cnt_in_2ea']

        df_포함여부_확인결과 = pd.DataFrame(li_포함여부, columns=li_컬럼명)
        # df_포함여부_확인결과 = df_포함여부_확인결과.sort_values('seq', ascending=False).reset_index(drop=True)

        # 결과 df 합치기
        if len(df_summary_기존.columns) == 1:
            df_summary_기존 = pd.DataFrame()
        df_summary = pd.concat([df_포함여부_확인결과, df_summary_기존], axis=0).drop_duplicates()
        df_summary = df_summary.sort_values('seq', ascending=False).reset_index(drop=True)

        # csv 저장
        s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_확률번호')
        s_파일명 = '확률번호포함확인_summary.csv'
        df_summary.to_csv(os.path.join(s_폴더명, s_파일명), index=False)

    def 포함확인_당첨번호위치(self):
        """ 6개확률 순서 중 당첨번호 위치 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_당첨번호위치')
        os.makedirs(s_폴더명, exist_ok=True)
        s_파일명 = '당첨번호위치_summary.csv'
        try:
            df_당첨위치 = pd.read_csv(os.path.join(s_폴더명, s_파일명), engine='python')
        except FileNotFoundError:
            df_당첨위치 = pd.DataFrame({'seq': [self.li_회차_n[0]]})
        ary_회차_완료 = df_당첨위치['seq'].values
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in ary_회차_완료]

        # 회차별 당첨번호 위치 확인
        for n_회차 in tqdm(li_회차_할거, desc='포함확인_당첨번호위치'):
            # n_회차 = 980              # 테스트용 임시
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['seq'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['date'].values[0].replace('.', '')
            ary_당첨_n = df_이력_회차.loc[:, 'win_1': 'win_6'].values[0]
            li_당첨 = [n for n in ary_당첨_n]
            s_당첨 = f'{ary_당첨_n[0]:02}_{ary_당첨_n[1]:02}_{ary_당첨_n[2]:02}' \
                   f'_{ary_당첨_n[3]:02}_{ary_당첨_n[4]:02}_{ary_당첨_n[5]:02}'

            # 확률 분석 결과 가져오기
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_확률_1개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            s_파일명 = f'확률예측_2개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_확률_2개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            s_파일명 = f'확률예측_6개번호_{n_회차}차_{s_추첨일}추첨.csv'
            try:
                df_확률_6개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            except FileNotFoundError:
                continue

            # 6개 숫자 통합컬럼 생성
            # df_확률_6개['nos'] = list(df_확률_6개.loc[:, 'no1': 'no6'].values)
            df_확률_6개['li_no'] = [[ary[0], ary[1], ary[2], ary[3], ary[4], ary[5]]
                                 for ary in df_확률_6개.loc[:, 'no1': 'no6'].values]
            df_확률_6개['nos'] = [f'{ary[0]:02}_{ary[1]:02}_{ary[2]:02}_{ary[3]:02}_{ary[4]:02}_{ary[5]:02}'
                               for ary in df_확률_6개.loc[:, 'no1': 'no6'].values]
            df_확률_6개['rank'] = df_확률_6개.index + 1

            # 당첨번호 위치 (전체)
            df_위치 = df_확률_6개[df_확률_6개['nos'] == s_당첨]
            if len(df_위치) == 0:
                n_위치 = 0
            else:
                n_위치 = df_위치['rank'].values[0]

            # 당첨번호 위치 (1개확률 1번~ 6번)
            li_위치 = [n_회차, s_추첨일, s_당첨, 'all', n_위치]
            for n_1개순위 in range(6):
                s_번호 = df_확률_1개['no'].values[n_1개순위]
                n_번호 = int(s_번호.replace('no', ''))
                df_확률_6개['match'] = df_확률_6개['li_no'].apply(lambda x: 1 if n_번호 in x else 0)
                df_위치 = df_확률_6개[df_확률_6개['match'] == 1].reset_index(drop=True)
                df_위치['rank'] = df_위치.index + 1
                df_위치 = df_위치[df_위치['nos'] == s_당첨]
                if len(df_위치) == 0:
                    n_위치 = 0
                else:
                    n_위치 = df_위치['rank'].values[0]
                li_위치.append(s_번호)
                li_위치.append(n_위치)

            # 당첨위치 결과 df 정리
            li_컬럼명 = ['seq', 'date', 'win', 'no_all', 'loc_all', 'no_1', 'loc_1', 'no_2', 'loc_2', 'no_3', 'loc_3',
                      'no_4', 'loc_4', 'no_5', 'loc_5', 'no_6', 'loc_6']
            df_당첨위치 = pd.DataFrame(li_위치).T
            df_당첨위치.columns = li_컬럼명

            # 기존 파일 불러와서 합치기
            s_파일명 = '당첨번호위치_summary.csv'
            try:
                df_당첨위치_기존 = pd.read_csv(os.path.join(s_폴더명, s_파일명), engine='python')
            except FileNotFoundError:
                df_당첨위치_기존 = pd.DataFrame()
            df_당첨위치 = pd.concat([df_당첨위치, df_당첨위치_기존], axis=0).drop_duplicates()
            df_당첨위치 = df_당첨위치.sort_values('seq', ascending=False).reset_index(drop=True)

            # csv 저장
            df_당첨위치.to_csv(os.path.join(s_폴더명, s_파일명), index=False)

    def 번호선정(self, s_선정로직):
        """ 확률 정보 읽어와서 번호선정로직에 따라 숫자 예측 및 결과 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_result_추가확인, s_선정로직)
        os.makedirs(s_폴더명, exist_ok=True)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                  if '확률예측_추첨결과_' in s_파일명 and '.csv' in s_파일명 and 'summary_' not in s_파일명]
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_회차_완료 = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in li_회차_완료]

        # 회차별 번호 선정 및 확인
        for n_회차 in tqdm(li_회차_할거, desc=s_선정로직):
            # n_회차 = 848 ############################################# 테스트용 #################
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['seq'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['date'].values[0].replace('.', '')
            ary_당첨_n = df_이력_회차.loc[:, 'win_1': 'win_6'].values[0]

            # 확률 분석 결과 가져오기
            dic_확률 = dict()
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            dic_확률['df_확률_1개'] = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            s_파일명 = f'확률예측_2개번호_{n_회차}차_{s_추첨일}추첨.csv'
            dic_확률['df_확률_2개'] = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
            if s_선정로직 in ['번호선정_2개번호_최빈수', '번호선정_2개번호_최빈수_중복제외', '번호선정_복합로직_최빈수_따라가기',
                          '번호선정_1개2개연계_최빈수연계']:
                s_파일명 = f'확률예측_6개번호_{n_회차}차_{s_추첨일}추첨.csv'
                try:
                    df_확률_6개 = pd.read_csv(os.path.join(self.folder_result_확률예측, s_파일명), engine='python')
                except FileNotFoundError:
                    continue
                for n in range(6):
                    df_확률_6개[f'no{n + 1}'] = df_확률_6개[f'no{n + 1}'].apply(lambda x: f'{x:02}')
                dic_확률['df_확률_6개'] = df_확률_6개

            # 번호 선정
            s_전략명, df_6개번호_5개세트 = None, None
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

            # 결과 확인
            li_li_번호조합 = [list(ary) for ary in df_6개번호_5개세트.values]
            li_당첨번호 = list(ary_당첨_n)
            li_결과_전체 = []
            for li_번호조합 in li_li_번호조합:
                li_당첨확인 = [1 if int(n_번호) in li_당첨번호 else 0 for n_번호 in li_번호조합]
                li_결과 = li_당첨확인 + [sum(li_당첨확인)] + list(li_번호조합) + list(li_당첨번호)
                li_결과_전체.append(li_결과)

            # 결과 df 생성
            li_컬럼명 = [f'win{n + 1}' for n in range(6)] + ['cnt_win'] \
                     + [f'no{n + 1}' for n in range(6)] + [f'win_{n + 1}' for n in range(6)]
            df_결과 = pd.DataFrame(li_결과_전체, columns=li_컬럼명)
            df_결과['seq'] = n_회차
            df_결과['date'] = s_추첨일

            # 상금 입력
            df_결과['award'] = df_결과['cnt_win'].apply(lambda x: 0 if x <= 2 else 5000 if x == 3 else 50000 if x == 4
            else 1000000 if x == 5 else 1000000000)

            # 컬럼 순서 정리
            li_컬럼명 = ['seq', 'date'] + [f'win{n + 1}' for n in range(6)] \
                     + ['cnt_win', 'award'] + [f'no{n + 1}' for n in range(6)] + [f'win_{n + 1}' for n in range(6)]
            df_결과 = df_결과.loc[:, li_컬럼명]

            # csv 저장
            n_갯수 = df_결과['cnt_win'].max()
            n_상금 = df_결과['award'].sum()
            s_폴더명 = os.path.join(self.folder_result_추가확인, s_선정로직)
            s_파일명 = f'확률예측_추첨결과_{n_회차}차_{s_추첨일}추첨_{n_갯수}개_{n_상금}원.csv'
            df_결과.to_csv(os.path.join(s_폴더명, s_파일명), index=False)

        # 당첨번호 포함 확인 csv 파일 읽어오기
        s_폴더명 = os.path.join(self.folder_result_추가확인, s_선정로직)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                  if '확률예측_추첨결과_' in s_파일명 and '.csv' in s_파일명 and '_summary' not in s_파일명]

        # 정보 정리
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_정보 = [[int(s_정보[2].replace('차', '')),
                  s_정보[3].replace('추첨', ''),
                  int(s_정보[4].replace('개', '')),
                  int(s_정보[5].replace('원.csv', ''))]
                 for s_정보 in li_정보]

        # df 생성
        li_컬럼명 = ['seq', 'date', 'win_cnt', 'award']
        df_정리 = pd.DataFrame(li_정보, columns=li_컬럼명)
        df_정리 = df_정리.sort_values('seq', ascending=False).reset_index(drop=True)

        # 기존 summary 파일 삭제
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명) if '_summary_' in s_파일명 and '.csv' in s_파일명]
        for s_파일명 in li_파일명:
            os.remove(os.path.join(s_폴더명, s_파일명))

        # csv 저장
        n_상금 = df_정리['award'].sum()
        s_파일명 = f'확률예측_추첨결과_summary_{n_상금:,}원.csv'
        df_정리.to_csv(os.path.join(s_폴더명, s_파일명), index=False)

    def 번호선정_결과한번에(self):
        """ 번호선정 폴더들에 있는 summary 파일 읽어와서 하나의 파일로 정리 후 csv 저장 """
        # summary 데이터 담을 list 생성
        li_df = []

        # 당첨번호 포함갯수 추가
        s_폴더명 = os.path.join(self.folder_result_추가확인, '포함확인_당첨번호')
        s_파일명 = '당첨번호포함여부_summary.csv'
        df_포함 = pd.read_csv(os.path.join(s_폴더명, s_파일명), engine='python')
        df_포함['번호포함갯수'] = df_포함['win_cnt']

        li_df.append(df_포함.loc[:, ['seq', 'date', '번호포함갯수']])

        # 번호선정 로직별 결과 추가
        li_폴더 = [s_폴더 for s_폴더 in os.listdir(self.folder_result_추가확인)
                 if '번호선정_' in s_폴더 and '.csv' not in s_폴더]

        # summary 파일 하나로 통합
        # li_df = []
        for s_폴더 in li_폴더:
            # summary 파일 읽어오기
            s_폴더명 = os.path.join(self.folder_result_추가확인, s_폴더)
            s_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명) if '_summary_' in s_파일명 and '.csv' in s_파일명][0]
            df_개별 = pd.read_csv(os.path.join(s_폴더명, s_파일명), engine='python')

            # # 구분자 집어넣기
            # if len(li_df) == 0:
            #     li_df.append(df_개별.loc[:, 'seq': 'date'])

            # 데이터 집어넣기
            s_로직 = s_폴더[5:]
            df_개별 = df_개별.sort_values('seq', ascending=True)
            df_개별[s_로직] = df_개별['award'].cumsum()
            df_개별 = df_개별.sort_values('seq', ascending=False)
            df_데이터 = df_개별.loc[:, ['win_cnt', 'award', s_로직]]
            df_데이터[s_로직] = df_데이터[s_로직].apply(lambda x: f'{x:,}')
            li_df.append(df_데이터)
        df_통합 = pd.concat(li_df, axis=1)

        # 파일 저장
        df_통합.to_csv(os.path.join(self.folder_result_추가확인, '번호선정_summary.csv'), encoding='cp949', index=False)


######################################################################################################################
if __name__ == '__main__':
    l = LottoAppendix()

    l.포함확인_당첨번호(n_번호확인갯수=30)
    l.포함확인_확률번호()
    l.포함확인_당첨번호위치()

    l.번호선정(s_선정로직='번호선정_2개번호_최빈수')
    l.번호선정(s_선정로직='번호선정_2개번호_최빈수_중복제외')
    l.번호선정(s_선정로직='번호선정_2개번호_따라가기')
    l.번호선정(s_선정로직='번호선정_2개번호_따라가기_확률반영')
    l.번호선정(s_선정로직='번호선정_복합로직_최빈수_따라가기')
    l.번호선정(s_선정로직='번호선정_1개2개연계_최빈수연계')

    l.번호선정_결과한번에()
