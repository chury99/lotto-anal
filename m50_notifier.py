import os
import sys
import json
import pandas as pd


# noinspection PyUnresolvedReferences,PyPep8Naming,PyProtectedMember
class Notifier:
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

        # dic_args 설정
        self.path_args = os.path.join(self.folder_run, 'dic_args.pkl')
        if 'dic_args.pkl' in os.listdir(self.folder_run):
            self.dic_args = pd.read_pickle(self.path_args)
        else:
            self.dic_args = dict()

        # 카카오 API 폴더 연결
        sys.path.append(dic_config['folder_kakao'])

        # log 기록
        self.make_log(f'### Notifier 구동 시작 ###')

    def 카톡으로보내기(self, s_선정로직):
        """ 선정된 번호조합 불러와서 카카오톡 메세지 발송 """
        # 전송할 회차 찾기 (번호 선정 폴더 내 가장 늦은 회차)
        s_폴더명 = os.path.join(self.folder_번호선정, f'번호선정로직_{s_선정로직[5:]}')
        li_파일명 = [s_파일명 for s_파일명 in os.listdir(s_폴더명)
                  if '확률예측_번호선정_' in s_파일명 and '.csv' in s_파일명 and 'summary_' not in s_파일명]

        li_정보 = [s_파일명.split(sep='_') for s_파일명 in li_파일명]
        n_회차 = max([int(s_정보[2].replace('차', '')) for s_정보 in li_정보])

        # 파일 찾기
        s_파일명 = [s_파일 for s_파일 in li_파일명 if f'_{n_회차}차_' in s_파일][0]
        s_추첨일 = s_파일명.split('_')[3].replace('추첨.csv', '')
        s_추첨일_dot = pd.Timestamp(s_추첨일).strftime('%y.%m.%d')

        # 메세지 작성
        df_선정번호 = pd.read_csv(os.path.join(s_폴더명, s_파일명), encoding='cp949')
        s_선정번호 = df_선정번호.to_string(index=False).replace('remark', ' rmk')
        s_기준정보 = f'# {n_회차}회 - {s_추첨일_dot}추첨 ({s_선정로직.split("_")[1]}) #'
        s_메세지 = s_기준정보 + '\n' + s_선정번호

        # 카톡 보내기
        import API_kakao
        k = API_kakao.KakaoAPI()
        result = k.send_message(s_user='알림봇', s_friend='여봉이', s_text=s_메세지)

        # log 기록
        if list(result.keys())[0].split('_')[0] == 'successful':
            self.make_log(f'### 카카오톡 메세지 발송 완료 ###')

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
    n = Notifier()
    n.카톡으로보내기(s_선정로직='번호선정_앙상블_결과종합')
