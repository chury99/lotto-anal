import os
import subprocess

# 피이썬 설정
path_파이썬 = os.path.join('C:\\Python311-64', 'python64.exe')

# 실행
p_10 = subprocess.Popen([path_파이썬, '10_collector.py'], shell=True)
# p_20 = subprocess.Popen([path_파이썬, '20_prob_maker.py'], shell=True)
# p_30 = subprocess.Popen([path_파이썬, '30_selector.py'], shell=True)
# p_40 = subprocess.Popen([path_파이썬, '40_verifier.py'], shell=True)
