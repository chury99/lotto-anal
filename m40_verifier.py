import os
import sys
import json
import pandas as pd
from tqdm import tqdm


# noinspection PyUnresolvedReferences,PyPep8Naming,PyProtectedMember
class Verifier:
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
        self.folder_결과확인 = os.path.join(self.folder_result, '결과확인')
        os.makedirs(self.folder_history, exist_ok=True)
        os.makedirs(self.folder_run, exist_ok=True)
        os.makedirs(self.folder_result, exist_ok=True)
        os.makedirs(self.folder_확률예측, exist_ok=True)
        os.makedirs(self.folder_번호선정, exist_ok=True)
        os.makedirs(self.folder_결과확인, exist_ok=True)

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
        self.make_log(f'### 결과 확인 시작 ###')

        # 이력 데이터 가져오기
        df_이력 = pd.read_csv(os.path.join(self.folder_history, 'lotto_history.csv'), encoding='cp949')
        self.df_이력 = df_이력.sort_values('회차', ascending=False).reset_index(drop=True)
        self.dic_회차_추첨일 = self.df_이력.set_index('회차').to_dict()['추첨일']

        # self 변수 정의 (회차 list)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(self.folder_확률예측)
                  if '확률예측_6개번호_' in s_파일명 and '.csv' in s_파일명]
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        self.li_회차_n = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]
        self.li_회차_n.sort()

    def 포함확인_당첨번호(self, n_번호확인갯수=30):
        """ 후보 숫자들이 당첨번호 안에 포함 되는지 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_당첨번호')
        os.makedirs(s_폴더명, exist_ok=True)
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                  if '당첨번호포함여부_' in s_파일명 and '.csv' in s_파일명 and 'summary' not in s_파일명]
        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        li_회차_완료 = [int(s_정보[1].replace('차', '')) for s_정보 in li_정보]
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in li_회차_완료]

        # 회차별 포함 여부 확인
        for n_회차 in tqdm(li_회차_할거, desc='포함확인_당첨번호'):
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['회차'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['추첨일'].values[0].replace('.', '')
            ary_당첨_n = df_이력_회차.loc[:, 'win_1': 'win_6'].values[0]

            # 예측번호 1개 확인
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측1개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
            ary_예측번호_1개_s = df_예측1개[:n_번호확인갯수]['no'].values
            li_예측번호_1개_n = [int(s[2:]) for s in ary_예측번호_1개_s]

            # 예측번호 2개 확인
            s_파일명 = f'확률예측_2개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측2개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
            ary_예측번호_2개_s = df_예측2개[:n_번호확인갯수]['no'].values
            li_예측번호_2개_n = [int(s[2:4]) for s in ary_예측번호_2개_s] + [int(s[5:]) for s in ary_예측번호_2개_s]

            # 예측번호 합치기 (중복 제거)
            li_예측번호_n = li_예측번호_1개_n + li_예측번호_2개_n
            li_예측번호_n = list(pd.Series(li_예측번호_n).unique())

            # df로 정리
            df_포함여부 = pd.DataFrame({'추출번호': li_예측번호_n})
            df_포함여부['포함여부'] = [1 if n in ary_당첨_n else 0 for n in li_예측번호_n]

            # 추가 정보 정리
            n_번호갯수 = len(df_포함여부)
            n_포함갯수 = df_포함여부['포함여부'].sum()

            # csv 저장
            s_파일명 = f'당첨번호포함여부_{n_회차}차_{s_추첨일}추첨_상위{n_번호확인갯수}개확인_{n_번호갯수}개추출_{n_포함갯수}개포함.csv'
            s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_당첨번호')
            df_포함여부.to_csv(os.path.join(s_폴더명, s_파일명), index=False, encoding='cp949')

        # 당첨번호 포함 확인 csv 파일 읽어오기
        s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_당첨번호')
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
        li_컬럼명 = ['회차', '추첨일', '확인갯수', '추출갯수', '포함갯수']
        df_정리 = pd.DataFrame(li_정보, columns=li_컬럼명)
        df_정리 = df_정리.sort_values('회차', ascending=False).reset_index(drop=True)

        # csv 저장
        s_파일명 = '당첨번호포함여부_summary.csv'
        df_정리.to_csv(os.path.join(s_폴더명, s_파일명), index=False, encoding='cp949')

        # 로그 기록
        self.make_log(f'# 당첨번호 포함여부 생성 완료 #')

    def 포함확인_확률번호(self):
        """ 확률 1개, 2개 번호 중 상위 6개 숫자가 당첨번호에 포함 되는지 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_확률번호')
        os.makedirs(s_폴더명, exist_ok=True)
        s_파일명 = '확률번호포함확인_summary.csv'
        try:
            df_summary_기존 = pd.read_csv(os.path.join(s_폴더명, s_파일명), encoding='cp949')
        except FileNotFoundError:
            df_summary_기존 = pd.DataFrame({'회차': [self.li_회차_n[0]]})
        ary_회차_완료 = df_summary_기존['회차'].values
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in ary_회차_완료]

        # 회차별 포함 여부 확인
        li_포함여부 = []
        for n_회차 in tqdm(li_회차_할거, desc='포함확인_확률번호'):
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['회차'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['추첨일'].values[0].replace('.', '')
            li_당첨_n = list(df_이력_회차.loc[:, '회차': 'win_6'].values[0])

            # 당첨번호포함여부 확인
            s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_당첨번호')
            li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                      if f'당첨번호포함여부_{n_회차}차_' in s_파일명 and '.csv' in s_파일명]
            n_당첨번호포함갯수 = int(li_파일명[0].split('_')[5].replace('개포함.csv', ''))

            # 예측번호 1개 확인
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_예측1개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')

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
            df_예측2개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')

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

            # 추첨 번호에 포함 여부 확인
            li_포함여부.append(li_결과)

        # 포함여부 확인 결과 df 정리
        li_컬럼명 = ['회차', '추첨일', 'win1', 'win2', 'win3', 'win4', 'win5', 'win6'] + ['포함갯수']\
                 + [f'1개번호_{n + 1}' for n in range(6)] + [f'1개포함_{n+1}' for n in range(6)] + ['1개포함갯수']\
                 + [f'2개번호_{n + 1}' for n in range(6)] + [f'2개포함_{n+1}' for n in range(6)] + ['2개포함갯수']

        df_포함여부_확인결과 = pd.DataFrame(li_포함여부, columns=li_컬럼명)

        # 결과 df 합치기
        if len(df_summary_기존.columns) == 1:
            df_summary_기존 = pd.DataFrame()
        df_summary = pd.concat([df_포함여부_확인결과, df_summary_기존], axis=0).drop_duplicates()
        df_summary = df_summary.sort_values('회차', ascending=False).reset_index(drop=True)

        # csv 저장
        s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_확률번호')
        s_파일명 = '확률번호포함확인_summary.csv'
        df_summary.to_csv(os.path.join(s_폴더명, s_파일명), index=False, encoding='cp949')

        # 로그 기록
        self.make_log(f'# 확률번호 포함확인 생성 완료 #')

    def 포함확인_당첨번호위치(self):
        """ 6개확률 순서 중 당첨번호 위치 확인 """
        # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
        s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_당첨번호위치')
        os.makedirs(s_폴더명, exist_ok=True)
        s_파일명 = '당첨번호위치_summary.csv'
        try:
            df_당첨위치 = pd.read_csv(os.path.join(s_폴더명, s_파일명), encoding='cp949')
        except FileNotFoundError:
            df_당첨위치 = pd.DataFrame({'회차': [self.li_회차_n[0]]})
        ary_회차_완료 = df_당첨위치['회차'].values
        li_회차_할거 = [n_회차 for n_회차 in self.li_회차_n if n_회차 not in ary_회차_완료]

        # 회차별 당첨번호 위치 확인
        for n_회차 in tqdm(li_회차_할거, desc='포함확인_당첨번호위치'):
            # 해당 회차 당첨번호 확인
            df_이력_회차 = self.df_이력[self.df_이력['회차'] == n_회차]
            if len(df_이력_회차) == 0:
                continue
            s_추첨일 = df_이력_회차['추첨일'].values[0].replace('.', '')
            ary_당첨_n = df_이력_회차.loc[:, 'win_1': 'win_6'].values[0]
            s_당첨 = f'{ary_당첨_n[0]:02}_{ary_당첨_n[1]:02}_{ary_당첨_n[2]:02}' \
                   f'_{ary_당첨_n[3]:02}_{ary_당첨_n[4]:02}_{ary_당첨_n[5]:02}'

            # 확률 분석 결과 가져오기
            s_파일명 = f'확률예측_1개번호_{n_회차}차_{s_추첨일}추첨.csv'
            df_확률_1개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
            s_파일명 = f'확률예측_6개번호_{n_회차}차_{s_추첨일}추첨.csv'
            try:
                df_확률_6개 = pd.read_csv(os.path.join(self.folder_확률예측, s_파일명), encoding='cp949')
            except FileNotFoundError:
                continue

            # 6개 숫자 통합 컬럼 생성
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
            li_컬럼명 = ['회차', '추첨일', '당첨번호', 'no_all', 'loc_all', 'no_1', 'loc_1', 'no_2', 'loc_2', 'no_3', 'loc_3',
                      'no_4', 'loc_4', 'no_5', 'loc_5', 'no_6', 'loc_6']
            df_당첨위치 = pd.DataFrame(li_위치).T
            df_당첨위치.columns = li_컬럼명

            # 기존 파일 불러와서 합치기
            s_파일명 = '당첨번호위치_summary.csv'
            try:
                df_당첨위치_기존 = pd.read_csv(os.path.join(s_폴더명, s_파일명), encoding='cp949')
            except FileNotFoundError:
                df_당첨위치_기존 = pd.DataFrame()
            df_당첨위치 = pd.concat([df_당첨위치, df_당첨위치_기존], axis=0).drop_duplicates()
            df_당첨위치 = df_당첨위치.sort_values('회차', ascending=False).reset_index(drop=True)

            # csv 저장
            df_당첨위치.to_csv(os.path.join(s_폴더명, s_파일명), index=False, encoding='cp949')

            # 로그 기록
            self.make_log(f'# 당첨번호 위치 생성 완료 #')

    def 결과확인(self):
        """ m30에서 선정한 번호를 가져와서 결과 확인 """
        # 선정 로직별 폴더 확인
        li_선정로직 = os.listdir(self.folder_번호선정)

        for s_선정로직 in li_선정로직:
            # 대상 회차 선정 (번호선정 완료된 회차 찾기)
            s_폴더_확인대상 = os.path.join(self.folder_번호선정, s_선정로직)
            li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더_확인대상)
                      if '확률예측_번호선정_' in s_파일명 and '.csv' in s_파일명 and 'summary_' not in s_파일명]
            li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
            li_회차_대상 = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]

            # 진행할 회차 선정 (완료된 파일 확인해서 새로운 회차만 진행)
            s_폴더_추첨결과 = os.path.join(self.folder_결과확인, s_선정로직)
            os.makedirs(s_폴더_추첨결과, exist_ok=True)
            li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더_추첨결과)
                      if '확률예측_추첨결과_' in s_파일명 and '.csv' in s_파일명 and 'summary_' not in s_파일명]
            li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
            li_회차_완료 = [int(s_정보[2].replace('차', '')) for s_정보 in li_정보]

            li_회차_할거 = [n_회차 for n_회차 in li_회차_대상 if n_회차 not in li_회차_완료]

            # 회차별 결과 확인
            for n_회차 in tqdm(li_회차_할거, desc=f'결과확인|{s_선정로직}'):
                # 해당 회차 정보 불러오기
                df_이력_회차 = self.df_이력[self.df_이력['회차'] == n_회차]
                if len(df_이력_회차) == 0:
                    continue
                s_추첨일 = df_이력_회차['추첨일'].values[0].replace('.', '')
                ary_당첨_n = df_이력_회차.loc[:, 'win_1': 'win_6'].values[0]

                # 로그 기록
                self.make_log(f'# {s_선정로직} 결과확인 - {n_회차}차_{s_추첨일}추첨 #')

                # 선정된 번호 불러오기
                df_선정번호 = pd.read_csv(os.path.join(s_폴더_확인대상, f'확률예측_번호선정_{n_회차}차_{s_추첨일}추첨.csv'))

                # 결과 확인
                li_li_번호조합 = [list(ary) for ary in df_선정번호.values]
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
                df_결과['회차'] = n_회차
                df_결과['추첨일'] = s_추첨일

                # 상금 입력
                df_결과['award'] = df_결과['cnt_win'].apply(lambda x: 0 if x <= 2 else
                                                                5000 if x == 3 else
                                                               50000 if x == 4 else
                                                             1000000 if x == 5 else
                                                          1000000000)

                # 컬럼 순서 정리
                li_컬럼명 = ['회차', '추첨일'] + [f'win{n + 1}' for n in range(6)] \
                         + ['cnt_win', 'award'] + [f'no{n + 1}' for n in range(6)] + [f'win_{n + 1}' for n in range(6)]
                df_결과 = df_결과.loc[:, li_컬럼명]

                # csv 저장
                n_갯수 = df_결과['cnt_win'].max()
                n_상금 = df_결과['award'].sum()
                s_파일명 = f'확률예측_추첨결과_{n_회차}차_{s_추첨일}추첨_{n_갯수}개_{n_상금}원.csv'
                df_결과.to_csv(os.path.join(s_폴더_추첨결과, s_파일명), index=False, encoding='cp949')

            # 당첨번호 포함 확인 csv 파일 읽어오기
            s_폴더명 = os.path.join(self.folder_결과확인, s_선정로직)
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
            li_컬럼명 = ['회차', '추첨일', 'win_cnt', 'award']
            df_정리 = pd.DataFrame(li_정보, columns=li_컬럼명)
            df_정리 = df_정리.sort_values('회차', ascending=False).reset_index(drop=True)

            # 기존 summary 파일 삭제
            li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명) if '_summary_' in s_파일명 and '.csv' in s_파일명]
            for s_파일명 in li_파일명:
                os.remove(os.path.join(s_폴더명, s_파일명))

            # csv 저장
            n_상금 = df_정리['award'].sum()
            s_파일명 = f'확률예측_추첨결과_summary_{n_상금:,}원.csv'
            df_정리.to_csv(os.path.join(s_폴더명, s_파일명), index=False, encoding='cp949')

    def 결과확인_한번에정리(self):
        """ 결과확인 폴더에 있는 있는 summary 파일 읽어와서 하나의 파일로 정리 후 csv 저장 """
        # summary 데이터 담을 list 생성
        li_df = []

        # 당첨번호 포함갯수 추가
        s_폴더명 = os.path.join(self.folder_결과확인, '포함확인_당첨번호')
        s_파일명 = '당첨번호포함여부_summary.csv'
        df_포함 = pd.read_csv(os.path.join(s_폴더명, s_파일명), encoding='cp949')
        df_포함['번호포함갯수'] = df_포함['포함갯수']

        li_df.append(df_포함.loc[:, ['회차', '추첨일', '번호포함갯수']])

        # 번호선정 로직별 결과 추가
        li_폴더 = [s_폴더 for s_폴더 in os.listdir(self.folder_결과확인)
                 if '번호선정로직_' in s_폴더 and '.csv' not in s_폴더]

        # summary 파일 하나로 통합
        for s_폴더 in li_폴더:
            # summary 파일 읽어오기
            s_폴더명 = os.path.join(self.folder_결과확인, s_폴더)
            s_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명) if '_summary_' in s_파일명 and '.csv' in s_파일명][0]
            df_개별 = pd.read_csv(os.path.join(s_폴더명, s_파일명), encoding='cp949')

            # # 구분자 집어넣기
            # if len(li_df) == 0:
            #     li_df.append(df_개별.loc[:, 'seq': 'date'])

            # 데이터 집어넣기
            s_로직 = s_폴더[5:]
            df_개별 = df_개별.sort_values('회차', ascending=True)
            df_개별[s_로직] = df_개별['award'].cumsum()
            df_개별 = df_개별.sort_values('회차', ascending=False)
            df_데이터 = df_개별.loc[:, ['win_cnt', 'award', s_로직]]
            df_데이터[s_로직] = df_데이터[s_로직].apply(lambda x: f'{x:,}')
            li_df.append(df_데이터)
        df_통합 = pd.concat(li_df, axis=1)

        # 파일 저장
        df_통합.to_csv(os.path.join(self.folder_결과확인, '번호선정_summary.csv'), index=False, encoding='cp949')

        # 로그 기록
        self.make_log(f'# 결과 정리 완료 #')

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


#######################################################################################################################
if __name__ == '__main__':
    v = Verifier()

    # 참고용 데이터 생성
    v.포함확인_당첨번호()
    v.포함확인_확률번호()
    v.포함확인_당첨번호위치()

    # 결과 확인 데이터 생성
    v.결과확인()
    v.결과확인_한번에정리()
