import logging
import numpy as np
import seaborn as sb
import datetime as dt
from pandas import DataFrame, Series, date_range
from matplotlib import pyplot as plt
from pycallgraphix.wrapper import register_method

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error

import concurrent.futures as futures

from .util import my_pretty_table
from .plot import my_lineplot

@register_method
def my_diff(
    data: DataFrame,
    yname: str,
    plot: bool = True,
    max_diff: int = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> None:
    """데이터의 정상성을 확인하고, 정상성을 충족하지 않을 경우 차분을 수행하여 정상성을 만족시킨다.
    반드시 데이터 프레임의 인덱스가 타임시리즈 데이터여야 한다.

    Args:
        data (DataFrame): 데이터 프레임
        yname (str): 차분을 수행할 데이터 컬럼명
        plot (bool, optional): 차분 결과를 그래프로 표시할지 여부. Defaults to True.
        max_diff (int, optional): 최대 차분 횟수. 지정되지 않을 경우 최대 반복. Defaults to None.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
    """
    df = data.copy()

    # 데이터 정상성 여부
    stationarity = False

    # 반복 수행 횟수
    count = 0

    # 데이터가 정상성을 충족하지 않는 동안 반복
    while not stationarity:
        if count == 0:
            print("=========== 원본 데이터 ===========")
        else:
            print("=========== %d차 차분 데이터 ===========" % count)
            # 차분 수행
            df = df.diff().dropna()

        if plot:
            my_lineplot(df=df, yname=yname, xname=df.index, figsize=figsize, dpi=dpi)

        # ADF Test
        ar = adfuller(df[yname])

        ardict = {
            "검정통계량(ADF Statistic)": [ar[0]],
            "유의수준(p-value)": [ar[1]],
            "최적차수(num of lags)": [ar[2]],
            "관측치 개수(num of observations)": [ar[3]],
        }

        for key, value in ar[4].items():
            ardict["기각값(Critical Values) %s" % key] = value

        stationarity = ar[1] <= 0.05
        ardict["데이터 정상성 여부"] = "정상" if stationarity else "비정상"

        ardf = DataFrame(ardict, index=["ADF Test"]).T
        my_pretty_table(ardf)

        # 반복회차 1 증가
        count += 1

        # 최대 차분 횟수가 지정되어 있고, 반복회차가 최대 차분 횟수에 도달하면 종료
        if max_diff and count == max_diff:
            break

    return df

@register_method
def my_rolling(
    data: Series,
    window: int,
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> Series:
    """이동평균을 계산한다. 반드시 index가 datetime 형식이어야 한다.

    Args:
        data (Series): 시리즈 데이터
        window (int): 이동평균 계산 기간
        plot (bool, optional): 이동평균 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
    Returns:
        Series: 이동평균 데이터
    """
    rolling = data.rolling(window=window).mean()

    if plot:
        df = DataFrame(
            {
                rolling.name: rolling,
            },
            index=rolling.index,
        )
        my_lineplot(
            df=df,
            yname=rolling.name,
            xname=df.index,
            figsize=figsize,
            dpi=dpi,
            callback=lambda ax: ax.set_title(f"Rolling (window={window})"),
        )

    return rolling

@register_method
def my_ewm(
    data: Series, span: int, plot: bool = True, figsize: tuple = (10, 5), dpi: int = 100
) -> Series:
    """지수가중이동평균을 계산한다.

    Args:
        data (Series): 시리즈 데이터
        span (int): 지수가중이동평균 계산 기간
        plot (bool, optional): 지수가중이동평균 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.

    Returns:
        Series: 지수가중이동평균 데이터
    """
    ewm = data.ewm(span=span).mean()

    if plot:
        df = DataFrame(
            {
                ewm.name: ewm,
            },
            index=ewm.index,
        )
        my_lineplot(
            df=df,
            yname=ewm.name,
            xname=df.index,
            figsize=figsize,
            dpi=dpi,
            callback=lambda ax: ax.set_title(f"Rolling (span={span})"),
        )

    return ewm

@register_method
def my_seasonal_decompose(
    data: Series,
    model: str = "additive",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
):
    """시계열 데이터를 계절적, 추세적, 불규칙적 성분으로 분해한다.

    Args:
        data (Series): 시리즈 데이터
        model (str, optional): 분해 모델. "additive" 또는 "multiplicative". Defaults to "additive".
        plot (bool, optional): 분해 결과 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
    """

    if model not in ["additive", "multiplicative"]:
        raise ValueError('model은 "additive"또는 "multiplicative"이어야 합니다.')

    sd = seasonal_decompose(data, model=model)

    sd_df = DataFrame(
        {
            "original": sd.observed,
            "trend": sd.trend,
            "seasonal": sd.seasonal,
            "resid": sd.resid,
        },
        index=data.index,
    )

    if plot:
        figure = sd.plot()
        figure.set_size_inches((figsize[0], figsize[1] * 4))
        figure.set_dpi(dpi)

        fig, ax1, ax2, ax3, ax4 = figure.get_children()

        ax1.set_ylabel("Original")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        plt.show()
        plt.close()

    return sd_df

@register_method
def my_timeseries_split(data: DataFrame, test_size: float = 0.2) -> tuple:
    """시계열 데이터를 학습 데이터와 테스트 데이터로 분할한다.

    Args:
        data (DataFrame): 시계열 데이터
        test_size (float, optional): 테스트 데이터 비율. Defaults to 0.2.

    Returns:
        tuple: (학습 데이터, 테스트 데이터)
    """
    train_size = 1 - test_size

    # 처음부터 70% 위치 전까지 분할
    train = data[: int(train_size * len(data))]

    # 70% 위치부터 끝까지 분할
    test = data[int(train_size * len(data)) :]

    return (train, test)

@register_method
def my_acf_plot(
    data: Series, figsize: tuple = (10, 5), dpi: int = 100, callback: any = None
):
    """ACF 그래프를 그린다.

    Args:
        data (Series): 시리즈 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
        callback (any, optional): 그래프에 추가할 콜백 함수. Defaults to None.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    plot_acf(data, ax=ax)
    ax.grid()

    if callback:
        callback(ax)

    plt.show()
    plt.close()

@register_method
def my_pacf_plot(
    data: Series, figsize: tuple = (10, 5), dpi: int = 100, callback: any = None
):
    """PACF 그래프를 그린다.

    Args:
        data (Series): 시리즈 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
        callback (any, optional): 그래프에 추가할 콜백 함수. Defaults to None.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    plot_pacf(data, ax=ax)
    ax.grid()

    if callback:
        callback(ax)

    plt.show()
    plt.close()

@register_method
def my_acf_pacf_plot(
    data: Series, figsize: tuple = (10, 5), dpi: int = 100, callback: any = None
):
    """ACF 그래프와 PACF 그래프를 그린다.

    Args:
        data (Series): 시리즈 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
        callback (any, optional): 그래프에 추가할 콜백 함수. Defaults to None.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2), dpi=dpi)

    plot_acf(data, ax=ax1)
    ax1.grid()

    plot_pacf(data, ax=ax2)
    ax2.grid()

    if callback:
        callback(ax1, ax2)

    plt.show()
    plt.close()

@register_method
def my_arima(
    train: Series,
    test: Series,
    auto: bool = False,
    p: int = 3,
    d: int = 3,
    q: int = 3,
    s: int = None,
    periods: int = 0,
    figsize: tuple = (15, 5),
    dpi: int = 100,
) -> ARIMA:
    """ARIMA 모델을 생성한다.

    Args:
        train (Series): 학습 데이터
        test (Series): 테스트 데이터
        auto (bool, optional): 최적의 ARIMA 모델을 찾을지 여부. Defaults to False.
        p (int, optional): AR 차수. Defaults to 0.
        d (int, optional): 차분 차수. Defaults to 0.
        q (int, optional): MA 차수. Defaults to 0.
        s (int, optional): 계절성 주기. Defaults to None.
        periods (int, optional): 예측 기간. Defaults to 0.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.

    Returns:
        ARIMA: ARIMA 모델
    """
    model = None

    if not auto:
        if s:
            model = ARIMA(train, order=(p, d, q), seasonal_order=(p, d, q, s))
        else:
            model = ARIMA(train, order=(p, d, q))

        fit = model.fit()
        print(fit.summary())

        start_index = 0
        end_index = len(train)
        test_pred = fit.predict(start=start_index, end=end_index)
        pred = fit.forecast(len(test) + periods)
    else:
        # 최적의 ARIMA 모델을 찾는다.
        if s:
            model = auto_arima(
                y=train,  # 모델링하려는 시계열 데이터 또는 배열
                start_p=0,  # p의 시작점
                max_p=p,  # p의 최대값
                d=d,  # 차분 횟수
                start_q=0,  # q의 시작점
                max_q=q,  # q의 최대값
                seasonal=True,  # 계절성 사용 여부
                m=s,  # 계절성 주기
                start_P=0,  # P의 시작점
                max_P=p,  # P의 최대값
                D=d,  # 계절성 차분 횟수
                start_Q=0,  # Q의 시작점
                max_Q=q,  # Q의 최대값
                trace=True,  # 학습 과정 표시 여부
            )
        else:
            model = auto_arima(
                y=train,  # 모델링하려는 시계열 데이터 또는 배열
                start_p=0,  # p의 시작점
                max_p=p,  # p의 최대값
                d=d,  # 차분 횟수
                start_q=0,  # q의 시작점
                max_q=q,  # q의 최대값
                seasonal=False,  # 계절성 사용 여부
                trace=True,  # 학습 과정 표시 여부
            )

        print(model.summary())
        pred = model.predict(n_periods=int(len(test)) + periods)
        pd = None

    # 예측 결과 그래프
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    sb.lineplot(data=train, x=train.index, y=train.columns[0], label="Train", ax=ax)
    sb.lineplot(data=test, x=test.index, y=test.columns[0], label="Test", ax=ax)

    if auto:
        sb.lineplot(
            x=pred.index, y=pred.values, label="Prediction", linestyle="--", ax=ax
        )
    else:
        sb.lineplot(
            x=test_pred.index, y=test_pred, label="Prediction", linestyle="--", ax=ax
        )
        sb.lineplot(x=pred.index, y=pred, label="Forecast", linestyle="--", ax=ax)

    ax.grid()
    ax.legend()

    plt.show()
    plt.close()

    return model

@register_method
def __prophet_execute(
    train: DataFrame,
    test: DataFrame = None,
    periods: int = 0,
    freq: str = "D",
    callback: any = None,
    **params,
):
    """Prophet 모델을 생성한다.

    Args:
        train (DataFrame): 훈련데이터
        test (DataFrame, optional): 검증데이터. Defaults to None.
        periods (int, optional): 예측기간. Defaults to 0.
        freq (str, optional): 예측주기(D,M,Y). Defaults to "D".
        callback (any, optional): 콜백함수. Defaults to None.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = Prophet(**params)
    if callback:
        callback(model)
    model.fit(train)

    size = 0 if test is None else len(test)
    freq = freq.upper()
    if freq not in ['D','M','Y']:
        freq = 'D'
    size = size + periods
    future = model.make_future_dataframe(periods=size , freq=freq)
    forecast = model.predict(future)

    if test is not None:
        pred = forecast[["ds", "yhat"]][-size :]
        score = np.sqrt(mean_squared_error(test["y"].values, pred["yhat"].values))
    else:
        pred = forecast[["ds", "yhat"]]
        score = np.sqrt(mean_squared_error(train["y"].values, pred["yhat"].values))

    return model, score, dict(params), forecast, pred

@register_method
def my_prophet(
    train: DataFrame,
    test: DataFrame = None,
    periods: int = 0,
    freq: str = "D",
    report: bool = True,
    print_forecast: bool = False,
    figsize=(20, 8),
    dpi: int = 200,
    callback: any = None,
    **params,
) -> DataFrame:
    """Prophet 모델을 생성한다.

    Args:
        train (DataFrame): 훈련 데이터
        test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        periods (int, optional): 예측기간. Defaults to 0.
        freq (str, optional): 예측주기(D,M,Y). Defaults to "D".
        report (bool, optional) : 독립변수 보고를 출력할지 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        callback (any, optional): 콜백함수. Defaults to None.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
    Returns:
        tuple: best_model, best_params, best_score
    """

    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    # ------------------------------------------------------
    # 분석모델 생성

    result = []
    processes = []

    if params:
        with futures.ThreadPoolExecutor() as executor:
            params = ParameterGrid(params)

            for p in params:
                processes.append(executor.submit(__prophet_execute, train=train, test=test, periods=periods, freq=freq, callback=callback, **p,))

            for p in futures.as_completed(processes):
                m, score, params, forecast, pred = p.result()
                result.append(
                    {
                        "model": m,
                        "params": params,
                        "score": score,
                        "forecast": forecast,
                        "pred": pred,
                    }
                )

    else:
        m, score, params, forecast, pred = __prophet_execute(train=train, test=test, periods=periods, freq=freq, callback=callback, **p)
        result.append(
            {
                "model": m,
                "params": params,
                "score": score,
                "forecast": forecast,
                "pred": pred,
            }
        )

    result_df = DataFrame(result).sort_values("score").reset_index(drop=True)
    best_model, best_params, best_score, best_forecast, best_pred = result_df.iloc[0]

    print_result = []
    for i, v in enumerate(result):
        item = v["params"]
        item["score"] = v["score"]
        print_result.append(item)

    my_pretty_table(
        DataFrame(print_result)
        .sort_values("score", ascending=True)
        .reset_index(drop=True)
    )

    if report:
        my_prophet_report(
            best_model, best_forecast, best_pred, test, print_forecast, figsize, dpi
        )
        
    return best_model, best_params, best_score, best_forecast, best_pred

@register_method
def my_prophet_report(
    model: Prophet,
    forecast: DataFrame,
    pred: DataFrame,
    test: DataFrame = None,
    print_forecast: bool = False,
    figsize: tuple = (20, 8),
    dpi: int = 100,
) -> DataFrame:
    """Prophet 모델 결과를 시각화한다.

    Args:
        model (Prophet): Prophet 모델
        forecast (DataFrame): 예측 결과
        pred (DataFrame): 예측 결과
        test (DataFrame, optional): 검증 데이터. Defaults to None.
        print_forecast (bool, optional): 예측 결과를 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)

    Returns:
        DataFrame: 독립변수 결과 보고
    """

    # ------------------------------------------------------
    # 결과 시각화
    fig = model.plot(forecast, figsize=figsize, xlabel="Date", ylabel="Value")
    fig.set_dpi(dpi)
    ax = fig.gca()
    add_changepoints_to_plot(ax, model, forecast)

    if test is not None:
        sb.lineplot(
            data=test,
            x="ds",
            y="y",
            color="#ff7f0e",
            linestyle="--",
            label="test",
            ax=ax,
        )
    plt.legend()
    plt.show()
    plt.close()

    fig = model.plot_components(forecast)
    fig.set_dpi(dpi)
    ax = fig.gca()
    
    plt.show()
    plt.close()

    # 예측 결과 테이블
    if print_forecast: my_pretty_table(forecast)

    if test is not None:
        yhat = forecast["yhat"].values[-len(test) :]
        y = test["y"].values

        result = {
            "평균절대오차(MAE)": mean_absolute_error(y, yhat),
            "평균제곱오차(MSE)": mean_squared_error(y, yhat),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y, yhat)),
        }

        my_pretty_table(DataFrame(result, index=["Prophet"]).T)

@register_method
def get_weekend_df(start: any, end: any = None) -> DataFrame:
    """주말 데이터 프레임을 생성한다.

    Args:
        start (any): 시작일
        end (any, optional): 종료일. Defaults to None.

    Returns:
        DataFrame: 주말 데이터 프레임
    """
    if end is None:
        end = dt.datetime.now()

    date = date_range(start, end)
    df = DataFrame({"date": date, "weekend": date.day_name()}).set_index("date")

    df["weekend"] = df["weekend"].apply(
        lambda x: 1 if x in ["Saturday", "Sunday"] else 0
    )

    df2 = df[df["weekend"] == 1]

    