# 경고메시지 출력 off
import warnings
warnings.filterwarnings(action='ignore')

# ---------------------
# 기본 라이브러리 참조
# ---------------------
# 파이썬 기본 시스템 모듈 -> 그래프 초기화 과정에서 OS종류에 따른 글꼴 선택을 위함
import sys

# 데이터 시각화 관련 모듈
from matplotlib import pyplot as plt

# -----------------
# 그래프 초기 설정
# -----------------
plt.rcParams["font.family"] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
plt.rcParams["font.size"] = 9
plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["axes.unicode_minus"] = False
