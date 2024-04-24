import cProfile
import joblib
from pycallgraphix.wrapper import register_method, MethodChart
from datetime import datetime as dt
import re
import requests
import contractions
import numpy as np
import nltk
from os.path import exists
from os import mkdir
from tabulate import tabulate
from pandas import DataFrame, read_excel, get_dummies, read_csv, Series, DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.stats import normaltest
from nltk.corpus import stopwords as stw

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys
from pca import pca
from matplotlib import pyplot as plt
from typing import Literal
from PIL import Image, ImageEnhance
import requests
from .core import *

# 형태소 분석 엔진 -> Okt
# from konlpy.tag import Okt

# 형태소 분석 엔진 -> Mecab
from konlpy.tag import Mecab

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


@register_method
def my_normalize_data(
    mean: float, std: float, size: int = 100, round: int = 2
) -> np.ndarray:
    """정규분포를 따르는 데이터를 생성한다.

    Args:
        mean (float): 평균
        std (float): 표준편차
        size (int, optional): 데이터 크기. Defaults to 1000.

    Returns:
        np.ndarray: 정규분포를 따르는 데이터
    """
    p = 0
    x = []
    while p < 0.05:
        x = np.random.normal(mean, std, size).round(round)
        _, p = normaltest(x)

    return x


@register_method
def my_normalize_df(
    means: list = [0, 0, 0],
    stds: list = [1, 1, 1],
    sizes: list = [100, 100, 100],
    rounds: int = 2,
) -> DataFrame:
    """정규분포를 따르는 데이터프레임을 생성한다.

    Args:
        means (list): 평균 목록
        stds (list): 표준편차 목록
        sizes (list, optional): 데이터 크기 목록. Defaults to [100, 100, 100].
        rounds (int, optional): 반올림 자리수. Defaults to 2.

    Returns:
        DataFrame: 정규분포를 따르는 데이터프레임
    """
    data = {}
    for i in range(0, len(means)):
        data[f"X{i+1}"] = my_normalize_data(means[i], stds[i], sizes[i], rounds)

    return DataFrame(data)


@register_method
def my_pretty_table(data: DataFrame) -> None:
    print(
        tabulate(
            data, headers="keys", tablefmt="psql", showindex=True, numalign="right"
        )
    )


@register_method
def my_read_excel(
    path: str,
    index_col: str = None,
    sheet_name: "str | int | list[int,str] | None" = 0,
    info: bool = True,
    categories: list = None,
    save: bool = False,
    timeindex: bool = False,
) -> DataFrame:
    """엑셀 파일을 데이터프레임으로 로드하고 정보를 출력한다.

    Args:
        path (str): 엑셀 파일의 경로(혹은 URL)
        index_col (str, optional): 인덱스 필드의 이름. Defaults to None.
        info (bool, optional): True일 경우 정보 출력. Defaults to True.
        timeindex (bool, optional): True일 경우 인덱스를 시계열로 설정. Defaults to False.
        categories (list, optional): 카테고리로 지정할 필드 목록. Defaults to None.
        save (bool, optional) : True일 경우 데이터프레임 저장. Defaults to False.
    Returns:
        DataFrame: 데이터프레임 객체
    """

    try:
        if index_col:
            data: DataFrame = read_excel(
                path, index_col=index_col, sheet_name=sheet_name
            )
        else:
            data: DataFrame = read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        print("\x1b[31m데이터를 로드하는데 실패했습니다.\x1b[0m")
        print(f"\x1b[31m{e}\x1b[0m")
        return None
    if save:
        if not exists("res"):
            mkdir("res")
        data.to_excel(f'./res/{path[1+path.rfind("/"):]}')
    if timeindex:
        data.index = DatetimeIndex(data.index)
    if categories:
        data = my_set_category(data, *categories)

    if info:
        print(data.info())

        print("\n데이터프레임 상위 5개 행")
        my_pretty_table(data.head())

        print("\n데이터프레임 하위 5개 행")
        my_pretty_table(data.tail())

        print("\n기술통계")
        desc = data.describe().T
        desc["nan"] = data.isnull().sum()
        my_pretty_table(desc)

        if categories:
            print("\n카테고리 정보")
            for c in categories:
                my_pretty_table(DataFrame({"count": data[c].value_counts()}))

    return data


@register_method
def my_read_csv(
    path: str,
    index_col: str = None,
    info: bool = True,
    categories: list = None,
    save: bool = False,
) -> DataFrame:
    """csv 파일을 데이터프레임으로 로드하고 정보를 출력한다.

    Args:
        path (str): 엑셀 파일의 경로(혹은 URL)
        index_col (str, optional): 인덱스 필드의 이름. Defaults to None.
        info (bool, optional): True일 경우 정보 출력. Defaults to True.
        categories (list, optional): 카테고리로 지정할 필드 목록. Defaults to None.
        save (bool, optional) : True일 경우 데이터프레임 저장. Defaults to False.
    Returns:
        DataFrame: 데이터프레임 객체
    """

    try:
        if index_col:
            data: DataFrame = read_csv(path, index_col=index_col)
        else:
            data: DataFrame = read_csv(path)
    except:
        try:
            if index_col:
                data: DataFrame = read_csv(
                    path,
                    index_col=index_col,
                    encoding="cp949",
                    encoding_errors="ignore",
                )
            else:
                data: DataFrame = read_csv(
                    path, encoding="cp949", encoding_errors="ignore"
                )
        except Exception as e:
            print("\x1b[31m데이터를 로드하는데 실패했습니다.\x1b[0m")
            print(f"\x1b[31m{e}\x1b[0m")
            return None
    if save:
        if not exists("res"):
            mkdir("res")
        data.to_excel(f'./res/{path[1+path.rfind("/"):]}')
    if categories:
        data = my_set_category(data, *categories)

    if info:
        print(data.info())

        print("\n데이터프레임 상위 5개 행")
        my_pretty_table(data.head())

        print("\n데이터프레임 하위 5개 행")
        my_pretty_table(data.tail())

        print("\n기술통계")
        desc = data.describe().T
        desc["nan"] = data.isnull().sum()
        my_pretty_table(desc)

    if categories:
        print("\n카테고리 정보")
        for c in categories:
            my_pretty_table(DataFrame(data[c].value_counts(), columns=[c]))

    return data


@register_method
def my_read_data(
    path: str,
    index_col: str = None,
    info: bool = True,
    categories: list = None,
    save: bool = False,
    timeindex: bool = False,
    sheet_name: any = 0,
) -> DataFrame:
    """파일을 데이터 프레임으로 로드하고 정보를 출력한다

    Args:
        path (str): 파일의 경로 (혹은 URL)
        index_col (str, optional) : 인덱스 필드의 이름. Defaults to None.
        info (bool, optional) : True일 경우 정보 출력. Defaults to True.
        save (bool, optional) : True일 경우 데이터프레임 저장. Defaults to False.
    Returns:
        DataFrame : 데이터프레임 객체
    """
    type = path[path.rfind(".") + 1 :]
    if type == "csv":
        return my_read_csv(
            path=path, index_col=index_col, info=info, categories=categories, save=save
        )
    elif type in ["xlsx", "xls"]:
        return my_read_excel(
            path=path,
            index_col=index_col,
            info=info,
            categories=categories,
            save=save,
            timeindex=timeindex,
            sheet_name=sheet_name,
        )


@register_method
def my_scaler(data: DataFrame, yname: str = None, method: str = "standard"):
    """데이터프레임의 연속형 변수에 대해 표준화를 수행한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.
        method (str, optional): 표준화 수행 방법['standard','minmax'] . Defaults to standard.
    """
    if method.lower() == "standard":
        return my_standard_scaler(data=data, yname=yname)
    elif method.lower() == "minmax":
        return my_minmax_scaler(data=data, yname=yname)
    else:
        raise Exception(f"\x1b[31m표준화방법 {method}가 존재하지 않습니다.\x1b[0m")


@register_method
def my_standard_scaler(data: DataFrame, yname: str = None) -> DataFrame:
    """데이터프레임의 연속형 변수에 대해 표준화를 수행한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.

    Returns:
        DataFrame: 표준화된 데이터프레임
    """
    # 원본 데이터 프레임 복사
    df = data.copy()

    # 종속변수만 별도로 분리
    if yname:
        y = df[yname]
        df = df.drop(yname, axis=1)

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 표준화 수행
    scaler = StandardScaler()
    std_df = DataFrame(scaler.fit_transform(df), index=data.index, columns=df.columns)

    # 분리했던 명목형 변수를 다시 결합
    if category_fields:
        std_df[category_fields] = cate

    # 분리했던 종속변수 결합
    if yname:
        std_df[yname] = y

    return std_df


@register_method
def my_minmax_scaler(data: DataFrame, yname: str = None) -> DataFrame:
    """데이터프레임의 연속형 변수에 대해 MinMax Scaling을 수행한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.

    Returns:
        DataFrame: 표준화된 데이터프레임
    """
    # 원본 데이터 프레임 복사
    df = data.copy()

    # 종속변수만 별도로 분리
    if yname:
        y = df[yname]
        df = df.drop(yname, axis=1)

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 표준화 수행
    scaler = MinMaxScaler()
    std_df = DataFrame(scaler.fit_transform(df), index=data.index, columns=df.columns)

    # 분리했던 명목형 변수를 다시 결합
    if category_fields:
        std_df[category_fields] = cate

    # 분리했던 종속 변수를 다시 결합
    if yname:
        std_df[yname] = y

    return std_df


@register_method
def my_train_test_split(
    data: any,
    yname: str = None,
    test_size: float = 0.2,
    random_state: int = get_random_state(),
    scalling: bool = False,
    save_path: str = None,
    load_path: str = None,
    ydata: any = None,
    categorical: bool = False,
) -> tuple:
    """데이터프레임을 학습용 데이터와 테스트용 데이터로 나눈다.

    Args:
        data (any): 데이터프레임 객체
        ydata (any, optional): 종속변수 데이터. Defaults to None.
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.
        test_size (float, optional): 검증 데이터의 비율(0~1). Defaults to 0.3.
        random_state (int, optional): 난수 시드. Defaults to 123.
        scalling (bool, optional): True일 경우 표준화를 수행한다. Defaults to False.
        save_path (str, optional): 스케일러 저장 경로. Defaults to None.
        load_path (str, optional): 스케일러 로드 경로. Defaults to None.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = None, None, None, None

    if yname is not None:
        if yname not in data.columns:
            raise Exception(f"\x1b[31m종속변수 {yname}가 존재하지 않습니다.\x1b[0m")

        x = data.drop(labels=yname, axis=1)
        y = data[yname]

        if categorical:
            res = np.zeros((len(y), len(y.unique())), dtype=int)
            res[np.arange(len(y)), y] = 1
            y = res

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
    elif ydata is not None:
        if categorical:
            res = np.zeros((len(ydata), len(ydata.unique())), dtype=int)
            res[np.arange(len(ydata)), ydata] = 1
            ydata = res

        x_train, x_test, y_train, y_test = train_test_split(
            data, ydata, test_size=test_size, random_state=random_state
        )
    else:
        x_train, x_test = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

    if scalling:
        if load_path:
            scaler = joblib.load(filename=load_path)
            x_train_std = scaler.transform(x_train)
            x_test_std = scaler.transform(x_test)
        else:
            scaler = StandardScaler()
            x_train_std = scaler.fit_transform(X=x_train)
            x_test_std = scaler.transform(x_test)

        x_train = DataFrame(
            data=x_train_std,
            index=x_train.index,
            columns=x_train.columns,
        )
        x_test = DataFrame(x_test_std, index=x_test.index, columns=x_test.columns)

        if save_path:
            joblib.dump(value=scaler, filename=save_path)

    if y_train is not None and y_test is not None:
        return x_train, x_test, y_train, y_test
    else:
        return x_train, x_test


@register_method
def my_set_category(data: DataFrame, *args: str) -> DataFrame:
    """카테고리 데이터를 설정한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        *args (str): 컬럼명 목록

    Returns:
        DataFrame: 카테고리 설정된 데이터프레임
    """
    df = data.copy()

    if not args:
        args = []
        for f in data.columns:
            if data[f].dtypes not in [
                "int",
                "int32",
                "int64",
                "float",
                "float32",
                "float64",
            ]:
                args.append(f)

    for k in args:
        df[k] = df[k].astype("category")

    return df


@register_method
def my_unmelt(
    data: DataFrame, id_vars: str = "class", value_vars: str = "values"
) -> DataFrame:
    """두 개의 컬럼으로 구성된 데이터프레임에서 하나는 명목형, 나머지는 연속형일 경우
    명목형 변수의 값에 따라 고유한 변수를 갖는 데이터프레임으로 변환한다.

    Args:
        data (DataFrame): 데이터프레임
        id_vars (str, optional): 명목형 변수의 컬럼명. Defaults to 'class'.
        value_vars (str, optional): 연속형 변수의 컬럼명. Defaults to 'values'.

    Returns:
        DataFrame: 변환된 데이터프레임
    """
    result = data.groupby(id_vars)[value_vars].apply(list)
    mydict = {}

    for i in result.index:
        mydict[i] = result[i]

    return DataFrame(mydict)


@register_method
def my_replace_missing_value(
    data: DataFrame, strategy: str = "mean", fill_value: str | int = None
) -> DataFrame:
    """결측치를 대체하여 데이터프레임을 재구성한다.

    Args:
        data (DataFrame): 데이터프레임
        strategy (["median", "mean", "most_frequent", "constant"], optional): 대체방법. Defaults to 'mean'.
        fill_value (str or numerical value): 상수로 대체할 경우 지정할 값.Defaults to '0'

    Returns:
        DataFrame: _description_
    """
    # 상수로 변환시 default값 0
    if strategy == "constant" and fill_value is None:
        fill_value = 0

    # 결측치 처리 규칙 생성
    imr = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)

    # 결측치 처리 규칙 적용 --> 2차원 배열로 반환됨
    df_imr = imr.fit_transform(data.values)

    # 2차원 배열을 데이터프레임으로 변환 후 리턴
    return DataFrame(df_imr, index=data.index, columns=data.columns)


@register_method
def my_drop_outliner(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 결측치로 변환한 후 모두 삭제한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 삭제된 데이터프레임
    """

    df = my_replace_outliner_to_nan(data, *fields)
    return df.dropna()


@register_method
def my_outlier_table(data: DataFrame, *fields: str) -> DataFrame:
    """데이터프레임의 사분위수와 결측치 경계값을 구한다.
    함수 호출 전 상자그림을 통해 결측치가 확인된 필드에 대해서만 처리하는 것이 좋다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: IQ
    """
    if not fields:
        fields = data.columns

    result = []
    for f in fields:
        # 숫자 타입이 아니라면 건너뜀
        if data[f].dtypes not in [
            "int",
            "int32",
            "int64",
            "float",
            "float32",
            "float64",
        ]:
            continue

        # 사분위수
        q1 = data[f].quantile(q=0.25)
        q2 = data[f].quantile(q=0.5)
        q3 = data[f].quantile(q=0.75)

        # 결측치 경계
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        iq = {
            "FIELD": f,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "IQR": iqr,
            "UP": up,
            "DOWN": down,
        }

        result.append(iq)

    return DataFrame(result).set_index("FIELD")


@register_method
def my_replace_outliner(data: DataFrame, *fields: str) -> DataFrame:
    """이상치 경계값을 넘어가는 데이터를 경계값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 경계값으로 대체된 데이터 프레임
    """

    # 원본 데이터 프레임 복사
    df = data.copy()

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 이상치 경계값을 구한다.
    outliner_table = my_outlier_table(df, *fields)

    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        df.loc[df[f] < outliner_table.loc[f, "DOWN"], f] = outliner_table.loc[f, "DOWN"]
        df.loc[df[f] > outliner_table.loc[f, "UP"], f] = outliner_table.loc[f, "UP"]

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df[category_fields] = cate

    return df


@register_method
def my_replace_outliner_to_nan(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 결측치로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 결측치로 대체된 데이터프레임
    """

    # 원본 데이터 프레임 복사
    df = data.copy()

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 이상치 경계값을 구한다.
    outliner_table = my_outlier_table(df, *fields)

    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        df.loc[df[f] < outliner_table.loc[f, "DOWN"], f] = np.nan
        df.loc[df[f] > outliner_table.loc[f, "UP"], f] = np.nan

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df[category_fields] = cate

    return df


@register_method
def my_replace_outliner_to_mean(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 평균값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 평균값으로 대체된 데이터프레임
    """
    # 원본 데이터 프레임 복사
    df = data.copy()

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 이상치를 결측치로 대체한다.
    if not fields:
        fields = df.columns

    df2 = my_replace_outliner_to_nan(df, *fields)

    # 결측치를 평균값으로 대체한다.
    df3 = my_replace_missing_value(df2, "mean")

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df3[category_fields] = cate

    return df3


@register_method
def my_dummies(data: DataFrame, *args: str) -> DataFrame:
    """명목형 변수를 더미 변수로 변환한다.

    Args:
        data (DataFrame): 데이터프레임
        *args (str): 명목형 컬럼 목록

    Returns:
        DataFrame: 더미 변수로 변환된 데이터프레임
    """
    if not args:
        args = [x for x in data.columns if data[x].dtypes == "category"]
    else:
        args = list(args)

    return get_dummies(data, columns=args, drop_first=True, dtype="int")


@register_method
def my_trend(x: any, y: any, degree: int = 2, value_count=100) -> tuple:
    """x, y 데이터에 대한 추세선을 구한다.

    Args:
        x : 산점도 그래프에 대한 x 데이터
        y : 산점도 그래프에 대한 y 데이터
        degree (int, optional): 추세선 방정식의 차수. Defaults to 2.
        value_count (int, optional): x 데이터의 범위 안에서 간격 수. Defaults to 100.

    Returns:
        tuple: (v_trend, t_trend)
    """
    # [ a, b, c ] ==> ax^2 + bx + c
    coeff = np.polyfit(x, y, degree)

    if type(x) == "list":
        minx = min(x)
        maxx = max(x)
    else:
        minx = x.min()
        maxx = x.max()

    v_trend = np.linspace(minx, maxx, value_count)

    t_trend = coeff[-1]
    for i in range(0, degree):
        t_trend += coeff[i] * v_trend ** (degree - i)

    return (v_trend, t_trend)


@register_method
def my_poly_features(
    data: DataFrame, columns: list = [], ignore: list = [], degree: int = 2
) -> DataFrame:
    """전달된 데이터프레임에 대해서 2차항을 추가한 새로온 데이터프레임을 리턴한다.

    Args:
        data (DataFrame): 원본 데이터 프레임
        columns (list, optional): 2차항을 생성할 필드 목록. 전달되지 않을 경우 전체 필드에 대해 처리 Default to [].
        ignore (list, optional): 2차항을 생성하지 않을 필드 목록. Default to [].
        degree (int, optional): 차수. Default to 2

    Returns:
        DataFrame: 2차항이 추가된 새로운 데이터 프레임
    """
    df = data.copy()

    if not columns:
        columns = df.columns

    ignore_df = None
    if ignore:
        ignore_df = df[ignore]
        df.drop(ignore, axis=1, inplace=True)
        columns = [c for c in columns if c not in set(ignore)]

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_fit = poly.fit_transform(df[columns])
    poly_df = DataFrame(poly_fit, columns=poly.get_feature_names_out(), index=df.index)

    df[poly_df.columns] = poly_df[poly_df.columns]

    if ignore_df is not None:
        df[ignore] = ignore_df

    return df


@register_method
def my_labelling(data: DataFrame, *fields) -> DataFrame:
    """명목형 변수를 라벨링한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 명목형 컬럼 목록

    Returns:
        DataFrame: 라벨링된 데이터프레임
    """
    df = data.copy()

    for f in fields:
        vc = sorted(list(df[f].unique()))
        label = {v: i for i, v in enumerate(vc)}
        df[f] = df[f].map(label).astype("int")

    return df


@register_method
def my_balance(xdata: DataFrame, ydata: Series, method: str = "smote") -> DataFrame:
    """불균형 데이터를 균형 데이터로 변환한다.

    Args:
        xdata (DataFrame): 독립변수 데이터 프레임
        ydata (Series): 종속변수 데이터 시리즈
        method (str, optional): 균형화 방법 [smote, over, under]. Defaults to 'smote'.

    Returns:
        DataFrame: _description_
    """

    if method == "smote":
        smote = SMOTE(random_state=get_random_state())
        xdata, ydata = smote.fit_resample(xdata, ydata)
    elif method == "over":
        ros = RandomOverSampler(random_state=get_random_state())
        xdata, ydata = ros.fit_resample(xdata, ydata)
    elif method == "under":
        rus = RandomUnderSampler(random_state=get_random_state())
        xdata, ydata = rus.fit_resample(xdata, ydata)
    else:
        raise Exception(
            f"\x1b[31m지원하지 않는 방법입니다.(smote, over, under중 하나를 지정해야 합니다.) ({method})\x1b[0m"
        )

    return xdata, ydata


@register_method
def my_vif_filter(
    data: DataFrame, yname: str = None, threshold: float = 10
) -> DataFrame:
    """독립변수 간 다중공선성을 검사하여 VIF가 threshold 이상인 변수를 제거한다.

    Args:
        data (DataFrame): 데이터프레임
        yname (str, optional): 종속변수 컬럼명. Defaults to None.
        threshold (float, optional): VIF 임계값. Defaults to 10.

    Returns:
        DataFrame: VIF가 threshold 이하인 변수만 남은 데이터프레임
    """
    df = data.copy()

    if yname:
        y = df[yname]
        df = df.drop(yname, axis=1)

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)
        elif len(df[f].unique()) <= 2:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # VIF 계산
    while True:
        xnames = list(df.columns)
        vif = {x: variance_inflation_factor(df, xnames.index(x)) for x in xnames}

        maxkey = max(vif, key=vif.get)

        if vif[maxkey] <= threshold:
            break

        df = df.drop(maxkey, axis=1)

    # 분리했던 명목형 변수를 다시 결합
    if category_fields:
        df[category_fields] = cate

    # 분리했던 종속 변수를 다시 결합
    if yname:
        df[yname] = y

    return df


@register_method
def my_pca(
    data: DataFrame,
    n_components: int | float = 0.95,
    standardize: bool = False,
    plot: bool = True,
    figsize: tuple = (15, 7),
    dpi: int = 100,
) -> DataFrame:
    """PCA를 수행하여 차원을 축소한다.

    Args:
        data (DataFrame): 데이터프레임
        n_components (int, optional): 축소할 차원 수[float : 설명할 비율, int : 표시할 차원의 수(주성분 갯수)]. Defaults to 0.95.
        standardize (bool, optional): True일 경우 표준화를 수행한다. Defaults to False.

    Returns:
        DataFrame: PCA를 수행한 데이터프레임
    """
    if standardize:
        df = my_standard_scaler(data)
    else:
        df = data.copy()

    model = pca(n_components=n_components, random_state=get_random_state())
    result = model.fit_transform(X=df)

    my_pretty_table(result["loadings"])
    my_pretty_table(result["topfeat"])

    if plot:
        fig, ax = model.biplot(figsize=figsize, fontsize=12, dpi=dpi)
        ax.set_title(ax.get_title(), fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        plt.show()
        plt.close()

        fig, ax = model.plot(figsize=figsize)
        fig.set_dpi(dpi)
        ax.set_title(ax.get_title(), fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        labels = ax.get_xticklabels()
        pc_labels = [f"PC{i+1}" for i in range(len(labels))]
        ax.set_xticklabels(pc_labels, fontsize=11, rotation=0)

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        plt.show()
        plt.close()

        plt.rcParams["font.family"] = (
            "AppleGothic" if sys.platform == "darwin" else "Malgun Gothic"
        )

    return result["PC"]


def my_trace() -> cProfile.Profile:
    profiler = cProfile.Profile()
    profiler.enable()

    methodchart = MethodChart()
    filename = "{0}.png".format(dt.now().strftime("%Y%m%d%H%M%S"))

    try:
        methodchart.make_graphviz_chart(time_resolution=3, filename=filename)
    except Exception as e:
        print(e)
        pass

    profiler.clear()
    profiler.disable()


@register_method
def tune_image(
    img: Image,
    mode: Literal["RGB", "color", "L", "gray"] = "RGB",
    size: tuple = None,
    color: float = None,
    contrast: int = None,
    brightness: float = None,
    sharpness: float = None,
) -> Image:
    """이미지를 튜닝한다.

    Args:
        img (Image): 이미지 객체
        mode (Literal['RGB', 'color', 'L', 'gray'], optional): 이미지 색상/흑백 모드
        size (tuple, optional): 이미지 크기. Defaults to None.
        color (float, optional): 이미지의 색상 균형을 조정한다. 0 부터 1 사이의 실수값으로 이미지의 색상을 조절 한다. 0 에 가까울 수록 색이 빠진 흑백에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 색이 더해진다. Defaults to None.
        contrast (int, optional): 이미지의 대비를 조정한다.  0에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해진다. Defaults to None.
        brightness (float, optional): 이미지의 밝기를 조정한다.  0에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해진다. Defaults to None.
        sharpness (float, optional): 이미지의 선명도를 조정한다. 0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해진다. Defaults to None.

    Returns:
        Image: 튜닝된 이미지
    """
    if mode:
        if mode == "color":
            mode = "RGB"
        elif mode == "gray":
            mode = "L"

        img = img.convert(mode=mode)

    if size:
        w = size[0] if size[0] > 0 else 0
        h = size[1] if size[1] > 0 else 0
        img = img.resize(size=(w, h))

    if color:
        if color < 0:
            color = 0
        img = ImageEnhance.Color(image=img).enhance(factor=color)

    if contrast:
        img = ImageEnhance.Contrast(image=img).enhance(
            factor=contrast if contrast > 0 else 0
        )

    if brightness:
        img = ImageEnhance.Brightness(image=img).enhance(
            factor=brightness if brightness > 0 else 0
        )

    if sharpness:
        img = ImageEnhance.Sharpness(image=img).enhance(
            factor=sharpness if sharpness > 0 else 0
        )

    img.array = np.array(img)

    return img


@register_method
def load_image(
    path: str,
    mode: Literal["RGB", "L"] = None,
    size: tuple = None,
    color: float = None,
    contrast: int = None,
    brightness: float = None,
    sharpness: float = None,
) -> Image:
    """이미지 파일을 로드한다. 필요한 경우 로드한 이미지에 대해 튜닝을 수행한다. 최종 로드된 이미지에 대한 배열 데이터를 array 속성에 저장한다.

    Args:
        path (str): 이미지 파일 경로
        mode (Literal['RGB', 'color', 'L', 'gray'], optional): 이미지 색상/흑백 모드
        size (tuple, optional): 이미지 크기. Defaults to None.
        color (float, optional): 이미지의 색상 균형을 조정한다. 0 부터 1 사이의 실수값으로 이미지의 색상을 조절 한다. 0 에 가까울 수록 색이 빠진 흑백에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 색이 더해진다. Defaults to None.
        contrast (int, optional): 이미지의 대비를 조정한다.  0에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해진다. Defaults to None.
        brightness (float, optional): 이미지의 밝기를 조정한다.  0에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해진다. Defaults to None.
        sharpness (float, optional): 이미지의 선명도를 조정한다. 0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해진다. Defaults to None.

    Returns:
        Image: 로드된 이미지
    """
    img = Image.open(fp=path)
    img = tune_image(
        img=img,
        mode=mode,
        size=size,
        color=color,
        contrast=contrast,
        brightness=brightness,
        sharpness=sharpness,
    )

    return img


@register_method
def my_stopwords(lang: str = "ko") -> list:
    stopwords = None

    if lang == "ko":
        session = requests.Session()
        session.headers.update(
            {
                "Referer": "",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
        )

        try:
            r = session.get("https://data.hossam.kr/tmdata/stopwords-ko.txt")

            # HTTP 상태값이 200이 아닌 경우는 에러로 간주한다.
            if r.status_code != 200:
                msg = "[%d Error] %s 에러가 발생함" % (r.status_code, r.reason)
                raise Exception(msg)

            r.encoding = "utf-8"
            stopwords = r.text.split("\n")
        except Exception as e:
            print(e)

    elif lang == "en":
        nltk.download("stopwords")
        stopwords = list(stw.words("english"))

    return stopwords


# -------------------------------------------------------------
@register_method
def my_text_morph(
    source: str, mode: str = "nouns", stopwords: list = None, dicpath: str = None
) -> list:
    """Mecab을 사용하여 텍스트를 형태소 분석한다.

    Args:
        source (str): 텍스트
        mode (str, optional): 분석 모드. Defaults to 'nouns'.

    Returns:
        list: 형태소 분석 결과
    """
    desc = None
    mecab = Mecab(dicpath=dicpath)

    if mode == "nouns":
        desc = mecab.nouns(phrase=source)
    elif mode == "morphs":
        desc = mecab.morphs(phrase=source)
    elif mode == "pos":
        desc = mecab.pos(phrase=source)
    else:
        desc = mecab.nouns(phrase=source)

    if stopwords:
        desc = [w for w in desc if w not in stopwords]

    return desc


# -------------------------------------------------------------
@register_method
def my_tokenizer(
    source: any, num_words: int = None, oov_token: str = "<OOV>", stopwords: list = None
):
    if type(source) == str:
        source = my_text_morph(source=source, stopwords=stopwords)

    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(source)

    return tokenizer

@register_method
def my_text_preprocessing(
    source: str,
    rm_abbr: bool = True,
    rm_email: bool = True,
    rm_html: bool = True,
    rm_url: bool = True,
    rm_num: bool = True,
    rm_special: bool = True,
    stopwords: list = None,
) -> str:
    """영문 텍스트를 전처리한다.

    Args:
        source (str): 텍스트
        rm_abbr (bool, optional): 약어 제거. Defaults to True.
        rm_email (bool, optional): 이메일 주소 제거. Defaults to True.
        rm_html (bool, optional): HTML 태그 제거. Defaults to True.
        rm_url (bool, optional): URL 주소 제거. Defaults to True.
        rm_num (bool, optional): 숫자 제거. Defaults to True.
        rm_special (bool, optional): 특수문자 제거. Defaults to True.
        stopwords (list, optional): 불용어 목록. Defaults to None.

    Returns:
        str: 전처리된 텍스트
    """
    # print(source)
    stopwords = set(stopwords)
    if stopwords is not None:
        source = " ".join([w for w in source.split() if w not in stopwords])
        # print(source)

    if rm_abbr:
        source = contractions.fix(source)
        # print(source)

    if rm_email:
        source = re.sub(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "", source
        )
        # print(source)

    if rm_html:
        source = re.sub(r"<[^>]*>", "", source)
        # print(source)

    if rm_url:
        source = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            source,
        )
        # print(source)

    if rm_num:
        source = re.sub(r"\b[0-9]+\b", "", source)
        # print(source)

    if rm_special:
        x = re.sub(r"[^\w ]+", "", source)
        source = " ".join(x.split())
        # print(source)

    return source


# -------------------------------------------------------------
@register_method
def my_text_data_preprocessing(
    data: DataFrame,
    fields: list = None,
    rm_abbr: bool = True,
    rm_email: bool = True,
    rm_html: bool = True,
    rm_url: bool = True,
    rm_num: bool = True,
    rm_special: bool = True,
    rm_stopwords: bool = True,
    stopwords: list = None,
) -> DataFrame:
    if not fields:
        fields = data.columns

    if type(fields) == str:
        fields = [fields]

    df = data.copy()

    for f in fields:
        df[f] = df[f].apply(
            lambda x: my_text_preprocessing(
                source=x,
                rm_abbr=rm_abbr,
                rm_email=rm_email,
                rm_html=rm_html,
                rm_url=rm_url,
                rm_num=rm_num,
                rm_special=rm_special,
                stopwords=stopwords,
            )
        )

    return df


# -------------------------------------------------------------
@register_method
def my_token_process(
    data: any,
    xname: str = None,
    yname: str = None,
    threshold: int = 10,
    num_words: int = None,
    max_word_count: int = None,
) -> DataFrame:
    # 훈련, 종속변수 분리
    x = None
    y = None

    if xname is not None:
        x = data[xname]
    else:
        x = data

    if yname is not None:
        y = data[yname]

    # 토큰화
    tokenizer = my_tokenizer(source=x)

    # 전체 단어의 수
    total_cnt = len(tokenizer.word_index)

    # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트할 값
    rare_cnt = 0

    # 훈련 데이터의 전체 단어 빈도수 총 합
    total_freq = 0

    # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
    rare_freq = 0

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    # --> [('one', 50324), ('reviewers', 500), ('mentioned', 1026), ('watching', 8909), ('oz', 256)]
    # --> key = 'one', value = 50324
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if value < threshold:
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print("단어 집합(vocabulary)의 크기 :", total_cnt)
    print("등장 빈도가 %s번 미만인 희귀 단어의 수: %s" % (threshold, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print(
        "전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100
    )

    # 자주 등장하는 단어 집합의 크기 구하기 -> 이 값이 첫 번째 학습층의 input 수가 된다.
    vocab_size = total_cnt - rare_cnt + 1
    print("단어 집합의 크기 :", vocab_size)

    # 최종 토큰화
    if num_words is None:
        num_words = vocab_size

    tokenizer2 = my_tokenizer(x, num_words=num_words)
    token_set = tokenizer2.texts_to_sequences(x)

    # 토큰화 결과 길이가 0인 항목의 index 찾기
    drop_target_index = []

    for i, v in enumerate(token_set):
        if len(v) < 1:
            drop_target_index.append(i)

    token_set2 = np.asarray(token_set, dtype="object")

    # 토큰 결과에서 해당 위치의 항목들을 삭제한다.
    fill_token_set = np.delete(token_set2, drop_target_index, axis=0)

    # 종속변수와 원래의 독립변수에서도 같은 위치의 항목들을 삭제해야 한다.
    future_set = np.delete(x, drop_target_index, axis=0)
    print("독립변수(텍스트) 데이터 수: ", len(fill_token_set))

    if y is not None:
        label_set = np.delete(y, drop_target_index, axis=0)
        print("종속변수(레이블) 데이터 수: ", len(label_set))

    # 문장별 단어 수 계산
    word_counts = []

    for s in fill_token_set:
        word_counts.append(len(s))

    if max_word_count is None:
        max_word_count = max(word_counts)

    pad_token_set = pad_sequences(fill_token_set, maxlen=max_word_count)
    pad_token_set_arr = [np.array(x, dtype="int") for x in pad_token_set]

    datadic = {}

    if y is not None:
        datadic[yname] = label_set

    if xname is not None:
        xname = "text"

    datadic[xname] = future_set
    datadic["count"] = word_counts
    datadic["token"] = fill_token_set
    datadic["pad_token"] = pad_token_set_arr

    df = DataFrame(data=datadic)

    return df, pad_token_set, vocab_size
