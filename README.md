# hossam-data-util

데이터 분석 유틸리티 


![Generic badge](https://img.shields.io/badge/version-0.0.1-critical.svg?style=flat-square&logo=appveyor) &nbsp;
[![The MIT License](https://img.shields.io/badge/license-MIT-orange.svg?style=flat-square&logo=appveyor)](http://opensource.org/licenses/MIT) &nbsp;
![Badge](https://img.shields.io/badge/Author-Lee%20KwangHo-blue.svg?style=flat-square&logo=appveyor) &nbsp;
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=appveyor) &nbsp;
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=appveyor) &nbsp;
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=appveyor) &nbsp;
![Scikit-learn](https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=appveyor)

## Hello World

이 자료는 메가스터디IT아카데미에서 진행중인 산대특 빅데이터 분석 과정의 수업 자료로 사용한 소스코드 입니다.

MIT 라이센스를 따릅니다.

## Installation

`pip` 명령으로 `setup.py` 파일이 있는 위치를 지정합니다.

### [1] Remote Repository

```shell
pip install --upgrade git+https://github.com/hyk2202/helper.git
```

or

```shell
pip install --upgrade git+ssh://git@github.com:hyk2202/helper.git
```



### [2] Local Repository

```shell
pip install --upgrade git+file:///path/to/your/git/project/
```

### [3] Local Directory

```shell
pip install --upgrade /path/to/your/project/
```


## Uninstallation

```shell
pip uninstall -y helper
```

## How to use

수업 중 적용되던 패키지 참조 코드가 아래와 같이 변경됩니다.

### 변경전

```Python
import sys
import os
work_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(work_path)

from helper.util import *
from helper.plot import *
from helper.analysis import *
from helper.classification import *
```

### 변경후

```Python
from helper.util import *
from helper.plot import *
from helper.analysis import *
from helper.classification import *
```


<!-- ## Documentation

[Documentation](https://leekh4232.github.io/hossam-data-helper/hossam) -->