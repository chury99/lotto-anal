import os
import subprocess

# 피이썬 설정
path_파이썬 = os.path.join('C:\\Python311-64', 'python64.exe')

# 실행
p_10 = subprocess.run([path_파이썬, 'm10_collector.py'], shell=True)
p_20 = subprocess.run([path_파이썬, 'm20_prob_maker.py'], shell=True)
p_30 = subprocess.run([path_파이썬, 'm30_selector.py'], shell=True)
p_40 = subprocess.run([path_파이썬, 'm40_verifier.py'], shell=True)
p_50 = subprocess.run([path_파이썬, 'm50_notifier.py'], shell=True)
