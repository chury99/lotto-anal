import os
import sys
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup


# noinspection PyUnresolvedReferences,PyPep8Naming
class Collector:
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
        os.makedirs(self.folder_history, exist_ok=True)
        os.makedirs(self.folder_run, exist_ok=True)
        os.makedirs(self.folder_result, exist_ok=True)

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
        self.make_log('### 초기 설정 완료 ###')

    def 추첨이력_업데이트(self):
        """ 회차별 추첨 이력 Update 후 history 폴더에 csv 저장 (네이버 크롤링) """
        # 로그 기록
        self.make_log('# 추첨이력 업데이트 시작 #')

        # 저장된 데이터 확인
        path_파일 = os.path.join(self.folder_history, 'lotto_history.csv')
        df_이력 = pd.read_csv(path_파일, encoding='cp949')

        # 기존 데이터 정보 확인
        df_이력 = df_이력.sort_values('추첨일', ascending=False).reset_index(drop=True).astype(str)
        n_회차_최종 = int(df_이력['회차'][0])
        s_추첨일_최종 = df_이력['추첨일'][0]
        dt_추첨일_최종 = pd.Timestamp(s_추첨일_최종)

        # 최종 데이터 이후 7일 경과 데이터 수집
        while pd.Timestamp(self.s_오늘) > dt_추첨일_최종 + pd.Timedelta(days=7):
            # 신규 데이터 크롤링
            n_회차_최종 += 1
            df_조회 = self._추첨번호_크롤링(n_회차=n_회차_최종)

            # 이전 데이터 병합
            df_이력 = pd.concat([df_이력, df_조회], axis=0)
            df_이력 = df_이력.drop_duplicates().sort_values('추첨일', ascending=False).reset_index(drop=True)
            n_회차_최종 = int(df_이력['회차'][0])
            s_추첨일_최종 = df_이력['추첨일'][0]
            dt_추첨일_최종 = pd.Timestamp(s_추첨일_최종)

            # 완료 후 csv 저장
            df_이력.to_csv(path_파일, index=False, encoding='cp949')

            # 로그 기록
            self.make_log(f'{n_회차_최종}회차 정보 업데이트 완료 - {s_추첨일_최종}')

        # 신규 데이터 미존재 시 메세지 출력
        if pd.Timestamp(self.s_오늘) <= dt_추첨일_최종 + pd.Timedelta(days=7):
            # 로그 기록
            self.make_log(f'신규 데이터 미존재 - 마지막 데이터 {n_회차_최종}회차 / {s_추첨일_최종}')

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
    def _추첨번호_크롤링(n_회차):
        """ 네이버 에서 추첨번호 크롤링 후 df 리턴 (조회 실패 시 'error' 리턴) """
        # 조회 요청
        url = f'https://search.naver.com/search.naver?query={n_회차}회로또'
        response = requests.get(url)

        # 조회 성공 시 날짜, 번호 수집
        if response.status_code == 200:
            # 뷰티풀 숩 설정
            soup = BeautifulSoup(response.text, 'html.parser')

            # 날짜 가져오기
            # tag_date = soup.select_one('#_lotto > div > div.lotto_tit > h3 > a > span')
            tag_date = soup.find('div', class_='select_tab')
            li_date = tag_date.text.split()
            s_추첨일 = li_date[1][1:-1]
            s_추첨일 = pd.Timestamp(s_추첨일).strftime('%Y.%m.%d')

            # 추첨번호 가져오기
            tag_numbers = soup.select('span.ball')
            li_추첨번호 = [tag_no.text for tag_no in tag_numbers]

        # 조회 실패 시 'error' 리턴
        else:
            s_추첨일 = 'error'
            li_추첨번호 = ['error'] * 7

        # df로 변환
        df = pd.DataFrame()
        df['회차'] = [str(n_회차)]
        df['추첨일'] = s_추첨일
        df['win_1'] = li_추첨번호[0]
        df['win_2'] = li_추첨번호[1]
        df['win_3'] = li_추첨번호[2]
        df['win_4'] = li_추첨번호[3]
        df['win_5'] = li_추첨번호[4]
        df['win_6'] = li_추첨번호[5]
        df['win_bonus'] = li_추첨번호[6]

        return df


#######################################################################################################################
if __name__ == '__main__':
    c = Collector()
    c.추첨이력_업데이트()
