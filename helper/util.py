from os.path import exists
from os import mkdir
import numpy as np
from tabulate import tabulate
from pandas import DataFrame, read_excel, get_dummies, read_csv, Series, DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from scipy.stats import normaltest

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


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


def my_pretty_table(data: DataFrame) -> None:
    print(
        tabulate(
            data, headers="keys", tablefmt="psql", showindex=True, numalign="right"
        )
    )


def my_read_excel(
    path: str,
    index_col: str = None,
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
            data: DataFrame = read_excel(path, index_col=index_col)
        else:
            data: DataFrame = read_excel(path)
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


def my_read_data(
    path: str,
    index_col: str = None,
    info: bool = True,
    categories: list = None,
    save: bool = False,
    timeindex: bool = False,
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
        )


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


def my_train_test_split(
    data: DataFrame,
    yname: str = "y",
    test_size: float = 0.2,
    random_state: int = 123,
    scalling: bool = False,
) -> tuple:
    """데이터프레임을 학습용 데이터와 테스트용 데이터로 나눈다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to 'y'.
        test_size (float, optional): 검증 데이터의 비율(0~1). Defaults to 0.3.
        random_state (int, optional): 난수 시드. Defaults to 123.
        scalling (bool, optional): True일 경우 표준화를 수행한다. Defaults to False.

    Returns:
        tuple: x_train, x_test, y_train, y_test

    """

    x = data.drop(yname, axis=1)
    y = data[yname]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    if scalling:
        scaler = StandardScaler()
        x_train = DataFrame(
            scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns
        )
        x_test = DataFrame(
            scaler.transform(x_test), index=x_test.index, columns=x_test.columns
        )

    return (x_train, x_test, y_train, y_test)


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


def my_replace_missing_value(
    data: DataFrame, strategy: str = "mean", fill_value: any = None
) -> DataFrame:
    """결측치를 대체하여 데이터프레임을 재구성한다.

    Args:
        data (DataFrame): 데이터프레임
        strategy (["median", "mean", "most_frequent", "constant"], optional): 대체방법. Defaults to 'mean'.
        fill_value (str or numerical value): 상수로 대체할 경우 지정할 값.Defaults to 'None'

    Returns:
        DataFrame: _description_
    """
    # 결측치 처리 규칙 생성
    imr = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)

    # 결측치 처리 규칙 적용 --> 2차원 배열로 반환됨
    df_imr = imr.fit_transform(data.values)

    # 2차원 배열을 데이터프레임으로 변환 후 리턴
    return DataFrame(df_imr, index=data.index, columns=data.columns)


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
        smote = SMOTE(random_state=0)
        xdata, ydata = smote.fit_resample(xdata, ydata)
    elif method == "over":
        ros = RandomOverSampler(random_state=0)
        xdata, ydata = ros.fit_resample(xdata, ydata)
    elif method == "under":
        rus = RandomUnderSampler(random_state=0)
        xdata, ydata = rus.fit_resample(xdata, ydata)
    else:
        raise Exception(
            f"\x1b[31m지원하지 않는 방법입니다.(smote, over, under중 하나를 지정해야 합니다.) ({method})\x1b[0m"
        )

    return xdata, ydata
