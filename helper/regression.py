import inspect
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import concurrent.futures as futures
from pycallgraphix.wrapper import register_method
from pandas import DataFrame, Series, concat

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    VotingRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from lightgbm import LGBMRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import t, f
from .util import my_pretty_table, my_trend, my_train_test_split
from .plot import my_residplot, my_qqplot, my_learing_curve, my_barplot
from .core import __ml, get_estimator, get_hyper_params

@register_method
def __my_regression(
    classname: any,
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = True,
    resid_test=True,
    deg: int = 1,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    est: any = None,
    **params,
) -> any:
    """회귀분석을 수행하고 결과를 출력한다.

    Args:
        classname (any): 회귀분석 추정기 (모델 객체)
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        est (any, optional): Voting, Bagging 앙상블 모델의 기본 추정기. Defaults to None.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        any: 회귀분석 모델
    """

    # ------------------------------------------------------
    # 분석모델 생성
    estimator = __ml(
        classname=classname,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        is_print=is_print,
        est=est,
        **params,
    )

    if estimator is None:
        print(f"\033[91m[{classname} 모델의 학습에 실패했습니다.\033[0m")
        return None

    # ------------------------------------------------------
    # 성능평가
    my_regression_result(
        estimator=estimator,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_curve=learning_curve,
        cv=cv,
        figsize=figsize,
        dpi=dpi,
        is_print=is_print,
    )

    # ------------------------------------------------------
    # 보고서 출력
    if report and is_print:
        print("")
        my_regression_report(
            estimator=estimator,
            x_train=estimator.x,
            y_train=estimator.y,
            x_test=sort,
            plot=plot,
            deg=deg,
            figsize=figsize,
            dpi=dpi,
        )

    # ------------------------------------------------------
    # 잔차 가정 확인
    if resid_test and is_print:
        print("\n\n[잔차의 가정 확인] ==============================")
        my_resid_test(
            x=estimator.x, y=estimator.y, y_pred=estimator.y_pred, figsize=figsize, dpi=dpi
        )

    return estimator

@register_method
def my_auto_linear_regression(df:DataFrame, yname:str, cv:int=5, learning_curve: bool = True, degree : int = 1, plot: bool = True, report=True, resid_test=False, figsize=(10, 4), dpi=150, sort: str = None,order: str = None,p_value_num:float=0.05) -> LinearRegression:
    """선형회귀분석을 수행하고 결과를 출력한다.

    Args:
        df (DataFrame) : 회귀분석을 수행할 데이터프레임.
        yname (str) : 종속변수
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        plot (bool, optional): 시각화 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 150.
        order (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        p_value_num (float, optional) : 회귀모형의 유의확률. Drfaults to 0.05
    Returns:
        LinearRegression: 회귀분석 모델
    """

    x_train, x_test, y_train, y_test = my_train_test_split(df, yname, test_size=0.2)
    
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성
    model = LinearRegression(n_jobs=-1) # n_jobs : 사용하는 cpu 코어의 개수 // -1은 최대치

    # 교차검증 설정
    if cv > 0:
        params = {}
        grid = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1)
        fit = grid.fit(x_train, y_train)
        model = fit.best_estimator_
        fit.best_params = fit.best_params_
        
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        
        # print("[교차검증]")
        # my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        # print("")

    fit = model.fit(x_train, y_train)
    x = x_test
    y = y_test
    y_pred = fit.predict(x)

    resid = y - y_pred

    # 절편과 계수를 하나의 배열로 결합
    params = np.append(fit.intercept_, fit.coef_)

    # 검증용 독립변수에 상수항 추가
    design_x = x.copy()
    design_x.insert(0, '상수', 1)

    dot = np.dot(design_x.T,design_x)   # 행렬곱
    inv = np.linalg.inv(dot)            # 역행렬
    dia = inv.diagonal()                # 대각원소

    # 제곱오차
    MSE = (sum((y-y_pred)**2)) / (len(design_x)-len(design_x.iloc[0]))

    se_b = np.sqrt(MSE * dia)           # 표준오차
    ts_b = params / se_b                # t값

    # 각 독립수에 대한 pvalue
    p_values = [2*(1-t.cdf(np.abs(i),(len(design_x)-len(design_x.iloc[0])))) for i in set(ts_b)]

    # VIF
    if len(x.columns) > 1:
        vif = [variance_inflation_factor(x, list(x.columns).index(v)) for  v in set(x.columns)]
    else:
        vif = 0

    # 표준화 계수
    train_df = x.copy()
    train_df[y.name] = y
    scaler = StandardScaler()
    std = scaler.fit_transform(train_df)
    std_df = DataFrame(std, columns=train_df.columns)
    std_x = std_df[xnames]
    std_y = std_df[yname]
    std_model = LinearRegression()
    std_fit = std_model.fit(std_x, std_y)
    beta = std_fit.coef_

    # 결과표 구성하기
    result_df = DataFrame({
        "종속변수": [yname] * len(xnames),
        "독립변수": xnames,
        "B(비표준화 계수)": np.round(params[1:], 4),
        "표준오차": np.round(se_b[1:], 3),
        "β(표준화 계수)": np.round(beta, 3),
        "t": np.round(ts_b[1:], 3),
        "유의확률": np.round(p_values[1:], 3),
        "VIF": vif,
    })
    if order:
        order = order.upper()
        if order == 'V':
            result_df.sort_values('VIF',inplace=True)
        elif  order == 'P':
            result_df.sort_values('유의확률',inplace=True)
        #result_df
    # my_pretty_table(result_df)
        
    resid = y - y_pred        # 잔차
    dw = durbin_watson(resid)               # 더빈 왓슨 통계량
    r2 = r2_score(y, y_pred)  # 결정계수(설명력)
    rowcount = len(x)                # 표본수
    featurecount = len(x.columns)    # 독립변수의 수

    # 보정된 결정계수
    adj_r2 = 1 - (1 - r2) * (rowcount-1) / (rowcount-featurecount-1)

    # f값
    f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

    # Prob (F-statistic)
    p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

    tpl = f"𝑅^2({r2:.3f}), Adj.𝑅^2({adj_r2:.3f}), F({f_statistic:.3f}), P-value({p:.4g}), Durbin-Watson({dw:.3f})"
    # print(tpl, end="\n\n")

    # 결과보고
    tpl = f"{yname}에 대하여 {','.join(xnames)}로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 유의{'하다' if p <= 0.05 else '하지 않다'}(F({len(x.columns)},{len(x.index)-len(x.columns)-1}) = {f_statistic:0.3f}, p {'<=' if p <= p_value_num else '>'} 0.05)."

    # # print(tpl, end = '\n\n')

    # 독립변수 보고
    for n in xnames:
        item = result_df[result_df['독립변수'] == n]
        coef = item['B(비표준화 계수)'].values[0]
        pvalue = item['유의확률'].values[0]

        s = f"{n}의 회귀계수는 {coef:0.3f}(p {'<=' if pvalue <= p_value_num else '>'} 0.05)로, {yname}에 대하여 {'유의미한' if pvalue <= p_value_num else '유의하지 않은'} 예측변인인 것으로 나타났다."

        # print(s)
        
    # print("")
    if result_df["VIF"].max() >= 10:
        # print('-'*50)
        # print('뺀 변수 :',result_df['독립변수'][result_df['VIF'].idxmax()])
        # print('-'*50)
        return my_auto_linear_regression(df.drop(result_df['독립변수'][result_df['VIF'].idxmax()],axis=1), yname, cv, degree,plot,report,resid_test, figsize, dpi, order,p_value_num )
    else:
        if result_df["유의확률"].max() >= p_value_num:
            # print('-'*50)
            # print('뺀 변수 :',result_df['독립변수'][result_df['유의확률'].idxmax()])
            # print('-'*50)
            return my_auto_linear_regression(df.drop(result_df['독립변수'][result_df['유의확률'].idxmax()],axis=1), yname,cv, degree,plot,report,resid_test, figsize, dpi, order,p_value_num )
    
    x_train, x_test, y_train, y_test = my_train_test_split(df, yname, test_size=0.2)
    
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성
    model = LinearRegression(n_jobs=-1) # n_jobs : 사용하는 cpu 코어의 개수 // -1은 최대치

    # 교차검증 설정
    if cv > 0:
        params = {}
        grid = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1)
        fit = grid.fit(x_train, y_train)
        model = fit.best_estimator_
        fit.best_params = fit.best_params_
        
        result_df = DataFrame(grid.cv_results_['params'])
        result_df['mean_test_score'] = grid.cv_results_['mean_test_score']
        
        print("[교차검증]")
        my_pretty_table(result_df.sort_values(by='mean_test_score', ascending=False))
        print("")

    fit = model.fit(x_train, y_train)
    x = x_test
    y = y_test
    y_pred = fit.predict(x)
    expr = "{yname} = ".format(yname=yname)

    for i, v in enumerate(xnames):
        expr += "%0.3f * %s + " % (fit.coef_[i], v)

    expr += "%0.3f" % fit.intercept_
    print("[회귀식]")
    print(expr, end="\n\n")
    resid = y - y_pred

    # 절편과 계수를 하나의 배열로 결합
    params = np.append(fit.intercept_, fit.coef_)

    # 검증용 독립변수에 상수항 추가
    design_x = x.copy()
    design_x.insert(0, '상수', 1)

    dot = np.dot(design_x.T,design_x)   # 행렬곱
    inv = np.linalg.inv(dot)            # 역행렬
    dia = inv.diagonal()                # 대각원소

    # 제곱오차
    MSE = (sum((y-y_pred)**2)) / (len(design_x)-len(design_x.iloc[0]))

    se_b = np.sqrt(MSE * dia)           # 표준오차
    ts_b = params / se_b                # t값

    # 각 독립수에 대한 pvalue
    p_values = [2*(1-t.cdf(np.abs(i),(len(design_x)-len(design_x.iloc[0])))) for i in set(ts_b)]

    # VIF
    if len(x.columns) > 1:
        vif = [variance_inflation_factor(x, list(x.columns).index(v)) for  v in set(x.columns)]
    else:
        vif = 0

    # 표준화 계수
    train_df = x.copy()
    train_df[y.name] = y
    scaler = StandardScaler()
    std = scaler.fit_transform(train_df)
    std_df = DataFrame(std, columns=train_df.columns)
    std_x = std_df[xnames]
    std_y = std_df[yname]
    std_model = LinearRegression()
    std_fit = std_model.fit(std_x, std_y)
    beta = std_fit.coef_

    # 결과표 구성하기
    result_df = DataFrame({
        "종속변수": [yname] * len(xnames),
        "독립변수": xnames,
        "B(비표준화 계수)": np.round(params[1:], 4),
        "표준오차": np.round(se_b[1:], 3),
        "β(표준화 계수)": np.round(beta, 3),
        "t": np.round(ts_b[1:], 3),
        "유의확률": np.round(p_values[1:], 3),
        "VIF": vif,
    })
    if order:
        order = order.upper()
        if order == 'V':
            result_df.sort_values('VIF',inplace=True)
        elif  order == 'P':
            result_df.sort_values('유의확률',inplace=True)
        # result_df
    my_pretty_table(result_df)
        
    resid = y - y_pred        # 잔차
    dw = durbin_watson(resid)               # 더빈 왓슨 통계량
    r2 = r2_score(y, y_pred)  # 결정계수(설명력)
    rowcount = len(x)                # 표본수
    featurecount = len(x.columns)    # 독립변수의 수

    # 보정된 결정계수
    adj_r2 = 1 - (1 - r2) * (rowcount-1) / (rowcount-featurecount-1)

    # f값
    f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

    # Prob (F-statistic)
    p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

    tpl = f"𝑅^2({r2:.3f}), Adj.𝑅^2({adj_r2:.3f}), F({f_statistic:.3f}), P-value({p:.4g}), Durbin-Watson({dw:.3f})"
    print(tpl, end="\n\n")

    # 결과보고
    tpl = f"{yname}에 대하여 {','.join(xnames)}로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 유의{'하다' if p <= 0.05 else '하지 않다'}(F({len(x.columns)},{len(x.index)-len(x.columns)-1}) = {f_statistic:0.3f}, p {'<=' if p <= p_value_num else '>'} 0.05)."

    print(tpl, end = '\n\n')

    # 독립변수 보고
    for n in xnames:
        item = result_df[result_df['독립변수'] == n]
        coef = item['B(비표준화 계수)'].values[0]
        pvalue = item['유의확률'].values[0]

        s = f"{n}의 회귀계수는 {coef:0.3f}(p {'<=' if pvalue <= p_value_num else '>'} 0.05)로, {yname}에 대하여 {'유의미한' if pvalue <= p_value_num else '유의하지 않은'} 예측변인인 것으로 나타났다."

        print(s)
        
    print("")
    return fit
    
@register_method
def my_linear_regression(x_train: DataFrame, y_train: Series, x_test: DataFrame = None, y_test: Series = None, cv: int = 5,  learning_curve: bool = True, deg:int = 1,degree : int = 1, plot: bool = True, is_print:bool = True,report=True, resid_test=False, figsize=(10, 4), dpi=150, sort: str = None,order: str = None,p_value_num:float=0.05, **params ) -> LinearRegression:
    """선형회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.
        order (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        p_value_num (float, optional) : 회귀모형의 유의확률. Drfaults to 0.05
    Returns:
        LinearRegression: 회귀분석 모델
    """
    xnames = x_train.columns
    yname = y_train.name
    size = len(xnames)

    # 분석모델 생성

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(LinearRegression)
        
    return __my_regression(
        classname=LinearRegression,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_ridge_regression(x_train: DataFrame, y_train: Series, x_test: DataFrame = None, y_test: Series = None, cv: int = 5, learning_curve: bool = True, report=False, plot: bool = False, degree: int = 1, resid_test=False, figsize=(10, 5), dpi: int = 100, sort: str = None, params: dict = {'alpha': [0.01, 0.1, 1, 10, 100]}) -> LinearRegression:
    """릿지회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        params (dict, optional): 하이퍼파라미터. Defaults to {'alpha': [0.01, 0.1, 1, 10, 100]}.
        
    Returns:
        Ridge: Ridge 모델
    """
    
    #------------------------------------------------------
    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=Ridge)

    return __my_regression(
        classname=Ridge,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_lasso_regression(    
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params)-> Lasso:
    """라쏘회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        degree (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        params (dict, optional): 하이퍼파라미터. Defaults to {'alpha': [0.01, 0.1, 1, 10, 100]}.
        
    Returns:
        Lasso: Lasso 모델
    """
    
    #------------------------------------------------------
    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=Lasso)

    return __my_regression(
        classname=Lasso,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def __regression_report_plot(ax: plt.Axes, x, y, xname, yname, y_pred, deg) -> None:
    if deg == 1:
        sb.regplot(x=x, y=y, ci=95, label="관측치", ax=ax)
        sb.regplot(x=x, y=y_pred, ci=0, label="추정치", ax=ax)
    else:
        sb.scatterplot(x=x, y=y, label="관측치", ax=ax)
        sb.scatterplot(x=x, y=y_pred, label="추정치", ax=ax)

        t1 = my_trend(x, y, degree=deg)
        sb.lineplot(
            x=t1[0], y=t1[1], color="blue", linestyle="--", label="관측치 추세선", ax=ax
        )

        t2 = my_trend(x, y_pred, degree=deg)
        sb.lineplot(
            x=t2[0], y=t2[1], color="red", linestyle="--", label="추정치 추세선", ax=ax
        )

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)

    ax.legend()
    ax.grid()

@register_method
def my_regression_result(
    estimator: any,
    x_train: DataFrame = None,
    y_train: Series = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    learning_curve: bool = True,
    cv: int = 10,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    is_print: bool = True,
) -> None:
    """회귀분석 결과를 출력한다.

    Args:
        estimator (any): 회귀분석 모델
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        cv (int, optional): 교차검증 횟수. Defaults to 10.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        is_print (bool, optional): 출력 여부. Defaults to True.
    """

    scores = []
    score_names = []

    if x_train is not None and y_train is not None:
        y_train_pred = estimator.predict(x_train)

        # 성능평가
        result = {
            "결정계수(R2)": r2_score(y_train, y_train_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_train, y_train_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_train, y_train_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "평균 절대 백분오차 비율(MAPE)": np.mean(
                np.abs((y_train - y_train_pred) / y_train) * 100
            ),
            "평균 비율 오차(MPE)": np.mean((y_train - y_train_pred) / y_train * 100),
        }

        scores.append(result)
        score_names.append("훈련데이터")

    if x_test is not None and y_test is not None:
        y_test_pred = estimator.predict(x_test)

        # 성능평가
        result = {
            "결정계수(R2)": r2_score(y_test, y_test_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_test, y_test_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_test, y_test_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "평균 절대 백분오차 비율(MAPE)": np.mean(
                np.abs((y_test - y_test_pred) / y_test) * 100
            ),
            "평균 비율 오차(MPE)": np.mean((y_test - y_test_pred) / y_test * 100),
        }

        scores.append(result)
        score_names.append("검증데이터")

    # 결과값을 모델 객체에 포함시킴
    estimator.scores = scores[-1]

    # ------------------------------------------------------
    if is_print:
        print("[회귀분석 성능평가]")
        result_df = DataFrame(scores, index=score_names)
        my_pretty_table(result_df.T)

        # 학습곡선
        if learning_curve:
            print("\n[학습곡선]")
            yname = y_train.name

            if x_test is not None and y_test is not None:
                y_df = concat([y_train, y_test])
                x_df = concat([x_train, x_test])
            else:
                y_df = y_train.copy()
                x_df = x_train.copy()

            x_df[yname] = y_df
            x_df.sort_index(inplace=True)

            if cv > 0:
                my_learing_curve(
                    estimator,
                    data=x_df,
                    yname=yname,
                    cv=cv,
                    scoring="RMSE",
                    figsize=figsize,
                    dpi=dpi,
                )
            else:
                my_learing_curve(
                    estimator,
                    data=x_df,
                    yname=yname,
                    scoring="RMSE",
                    figsize=figsize,
                    dpi=dpi,
                )

        if estimator.__class__.__name__ == "XGBRegressor":
            print("\n[변수 중요도]")
            my_plot_importance(estimator=estimator)

            feature_important = estimator.get_booster().get_score(
                importance_type="weight"
            )
            keys = list(feature_important.keys())
            values = list(feature_important.values())

            data = DataFrame(data=values, index=keys, columns=["score"]).sort_values(
                by="score", ascending=False
            )

            data["rate"] = data["score"] / data["score"].sum()
            data["cumsum"] = data["rate"].cumsum()

            my_pretty_table(data)

            # print("\n[TREE]")
            # my_xgb_tree(booster=estimator)


@register_method
def my_regression_report(
    estimator: any,
    x_train: DataFrame = None,
    y_train: Series = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    sort: str = None,
    plot: bool = False,
    deg: int = 1,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> None:
    """선형회귀분석 결과를 보고한다.

    Args:
        estimator (LinearRegression): 선형회귀 객체
        x_train (DataFrame, optional): 훈련 데이터의 독립변수. Defaults to None.
        y_train (Series, optional): 훈련 데이터의 종속변수. Defaults to None.
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        sort (str, optional): 정렬 기준 (v, p). Defaults to None.
        plot (bool, optional): 시각화 여부. Defaults to False.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
    """

    # ------------------------------------------------------
    # 회귀식

    if x_test is not None and y_test is not None:
        x = x_test.copy()
        y = y_test.copy()
    else:
        x = x_train.copy()
        y = y_train.copy()

    xnames = x.columns
    yname = y.name
    y_pred = estimator.predict(x)

    if estimator.__class__.__name__ in ["LinearRegression", "Lasso", "Ridge"]:
        expr = "{yname} = ".format(yname=yname)

        for i, v in enumerate(xnames):
            expr += "%0.3f * %s + " % (estimator.coef_[i], v)

        expr += "%0.3f" % estimator.intercept_
        print("[회귀식]")
        print(expr, end="\n\n")

        print("[독립변수보고]")

        if x is None and y is None:
            x = estimator.x
            y = estimator.y

        # 잔차
        resid = y - y_pred

        # 절편과 계수를 하나의 배열로 결합
        params = np.append(estimator.intercept_, estimator.coef_)

        # 검증용 독립변수에 상수항 추가
        design_x = x.copy()
        design_x.insert(0, "상수", 1)

        dot = np.dot(design_x.T, design_x)  # 행렬곱
        inv = np.linalg.inv(dot)  # 역행렬
        dia = inv.diagonal()  # 대각원소

        # 제곱오차
        MSE = (sum((y - y_pred) ** 2)) / (len(design_x) - len(design_x.iloc[0]))

        se_b = np.sqrt(MSE * dia)  # 표준오차
        ts_b = params / se_b  # t값

        # 각 독립수에 대한 pvalue
        p_values = [
            2 * (1 - t.cdf(np.abs(i), (len(design_x) - len(design_x.iloc[0]))))
            for i in ts_b
        ]

        # VIF
        if len(x.columns) > 1:
            vif = [
                variance_inflation_factor(x, list(x.columns).index(v))
                for i, v in enumerate(x.columns)
            ]
        else:
            vif = 0

        # 표준화 계수
        train_df = x.copy()
        train_df[y.name] = y
        scaler = StandardScaler()
        std = scaler.fit_transform(train_df)
        std_df = DataFrame(std, columns=train_df.columns)
        std_x = std_df[xnames]
        std_y = std_df[yname]
        std_estimator = LinearRegression(n_jobs=-1)
        std_estimator.fit(std_x, std_y)
        beta = std_estimator.coef_

        # 결과표 구성하기
        result_df = DataFrame(
            {
                "종속변수": [yname] * len(xnames),
                "독립변수": xnames,
                "B(비표준화 계수)": np.round(params[1:], 4),
                "표준오차": np.round(se_b[1:], 3),
                "β(표준화 계수)": np.round(beta, 3),
                "t": np.round(ts_b[1:], 3),
                "유의확률": np.round(p_values[1:], 3),
                "VIF": vif,
            }
        )

        if sort:
            if sort.upper() == "V":
                result_df = result_df.sort_values("VIF", ascending=False).reset_index()
            elif sort.upper() == "P":
                result_df = result_df.sort_values(
                    "유의확률", ascending=False
                ).reset_index()

        # result_df
        my_pretty_table(result_df)
        print("")

        resid = y - y_pred  # 잔차
        dw = durbin_watson(resid)  # 더빈 왓슨 통계량
        r2 = r2_score(y, y_pred)  # 결정계수(설명력)
        rowcount = len(x)  # 표본수
        featurecount = len(x.columns)  # 독립변수의 수

        # 보정된 결정계수
        adj_r2 = 1 - (1 - r2) * (rowcount - 1) / (rowcount - featurecount - 1)

        # f값
        f_statistic = (r2 / featurecount) / ((1 - r2) / (rowcount - featurecount - 1))

        # Prob (F-statistic)
        p = 1 - f.cdf(f_statistic, featurecount, rowcount - featurecount - 1)

        tpl = "𝑅^2(%.3f), Adj.𝑅^2(%.3f), F(%.3f), P-value(%.4g), Durbin-Watson(%.3f)"
        print(tpl % (r2, adj_r2, f_statistic, p, dw), end="\n\n")

        # 결과보고
        tpl = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과,\n이 회귀모형은 통계적으로 %s(F(%s,%s) = %0.3f, p %s 0.05)."

        result_str = tpl % (
            yname,
            ",".join(xnames),
            "유의하다" if p <= 0.05 else "유의하지 않다",
            len(x.columns),
            len(x.index) - len(x.columns) - 1,
            f_statistic,
            "<=" if p <= 0.05 else ">",
        )

        print(result_str, end="\n\n")

        # 독립변수 보고
        for n in xnames:
            item = result_df[result_df["독립변수"] == n]
            coef = item["B(비표준화 계수)"].values[0]
            pvalue = item["유의확률"].values[0]

            s = "%s의 회귀계수는 %0.3f(p %s 0.05)로, %s에 대하여 %s."
            k = s % (
                n,
                coef,
                "<=" if pvalue <= 0.05 else ">",
                yname,
                (
                    "유의미한 예측변인인 것으로 나타났다"
                    if pvalue <= 0.05
                    else "유의하지 않은 예측변인인 것으로 나타났다"
                ),
            )

            print(k)

        # 도출된 결과를 회귀모델 객체에 포함시킴 --> 객체 타입의 파라미터는 참조변수로 전달되므로 fit 객체에 포함된 결과값들은 이 함수 외부에서도 사용 가능하다.
        estimator.r2 = r2
        estimator.adj_r2 = adj_r2
        estimator.f_statistic = f_statistic
        estimator.p = p
        estimator.dw = dw

    else:
        # VIF
        if len(x.columns) > 1:
            vif = [
                variance_inflation_factor(x, list(x.columns).index(v))
                for i, v in enumerate(x.columns)
            ]
        else:
            vif = 0

        # 결과표 구성하기
        result_df = DataFrame(
            {
                "종속변수": [yname] * len(xnames),
                "독립변수": xnames,
                "VIF": vif,
            }
        )

        if sort:
            if sort.upper() == "V":
                result_df = result_df.sort_values("VIF", ascending=False).reset_index()

        # result_df
        my_pretty_table(result_df)
        print("")

    # 시각화
    if plot:
        size = len(xnames)
        cols = 2
        rows = (size + cols - 1) // cols

        fig, ax = plt.subplots(
            nrows=rows,
            ncols=cols,
            squeeze=False,
            figsize=(figsize[0] * cols, figsize[1] * rows),
            dpi=dpi,
        )

        fig.subplots_adjust(wspace=0.1, hspace=0.3)

        with futures.ThreadPoolExecutor() as executor:
            for i, v in enumerate(xnames):
                r = i // cols
                c = i % cols

                executor.submit(
                    __regression_report_plot,
                    ax=ax[r, c],
                    x=x[v],
                    y=y,
                    xname=v,
                    yname=yname,
                    y_pred=y_pred,
                    deg=deg,
                )

        plt.show()
        plt.close()

@register_method
def my_resid_normality(y: Series, y_pred: Series) -> None:
    """MSE값을 이용하여 잔차의 정규성 가정을 확인한다.

    Args:
        y (Series): 종속변수
        y_pred (Series): 예측값
    """
    mse = mean_squared_error(y, y_pred)
    resid = y - y_pred
    mse_sq = np.sqrt(mse)

    r1 = resid[ (resid > -mse_sq) & (resid < mse_sq)].count() / resid.count() * 100
    r2 = resid[ (resid > -2*mse_sq) & (resid < 2*mse_sq)].count() / resid.count() * 100
    r3 = resid[ (resid > -3*mse_sq) & (resid < 3*mse_sq)].count() / resid.count() * 100

    mse_r = [r1, r2, r3]
    
    print(f"루트 1MSE 구간에 포함된 잔차 비율: {r1:1.2f}% ({r1-68})")
    print(f"루트 2MSE 구간에 포함된 잔차 비율: {r2:1.2f}% ({r2-95})")
    print(f"루트 3MSE 구간에 포함된 잔차 비율: {r3:1.2f}% ({r3-99})")
    
    normality = r1 >= 68 and r2 >= 95 and r3 >= 99
    print(f"잔차의 정규성 가정 충족 여부: {normality}")

@register_method
def my_resid_equal_var(x: DataFrame, y: Series, y_pred: Series, p_value_num:float =0.05) -> None:
    """잔차의 등분산성 가정을 확인한다.

    Args:
        x (DataFrame): 독립변수
        y (Series): 종속변수
        y_pred (Series): 예측값
        p_value_num(float) : 유의확률
    """
    # 독립변수 데이터 프레임 복사
    x_copy = x.copy()
    
    # 상수항 추가
    x_copy.insert(0, "const", 1)
    
    # 잔차 구하기
    resid = y - y_pred
    
    # 등분산성 검정
    bs_result = het_breuschpagan(resid, x_copy)
    bs_result_df = DataFrame(bs_result, columns=['values'], index=['statistic', 'p-value', 'f-value', 'f p-value'])

    print(f"잔차의 등분산성 가정 충족 여부: {bs_result[1] > p_value_num}")
    my_pretty_table(bs_result_df)

@register_method
def my_resid_independence(y: Series, y_pred: Series) -> None:
    """잔차의 독립성 가정을 확인한다.

    Args:
        y (Series): 종속변수
        y_pred (Series): 예측값
    """
    dw = durbin_watson(y - y_pred)
    print(f"Durbin-Watson: {dw}, 잔차의 독립성 가정 만족 여부: {1.5 < dw < 2.5}")
    
@register_method    
def my_resid_test(x: DataFrame, y: Series, y_pred: Series, figsize: tuple=(10, 4), dpi: int=150, p_value_num:float = 0.05) -> None:
    """잔차의 가정을 확인한다.

    Args:
        x (Series): 독립변수
        y (Series): 종속변수
        y_pred (Series): 예측값
        p_value_num(float) : 유의확률
    """

    # 잔차 생성
    resid = y - y_pred
    
    print("[잔차의 선형성 가정]")
    my_residplot(y, y_pred, lowess=True, figsize=figsize, dpi=dpi)
    
    print("\n[잔차의 정규성 가정]")
    my_qqplot(y, figsize=figsize, dpi=dpi)
    my_residplot(y, y_pred, mse=True, figsize=figsize, dpi=dpi)
    my_resid_normality(y, y_pred)
    
    print("\n[잔차의 등분산성 가정]")
    my_resid_equal_var(x, y, y_pred, p_value_num)
    
    print("\n[잔차의 독립성 가정]")
    my_resid_independence(y, y_pred)

@register_method
def my_ridge_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> Ridge:
    """릿지회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        Ridge: Ridge 모델
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = {"alpha": [0.01, 0.1, 1, 10, 100]}

    return __my_regression(
        classname=Ridge,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_lasso_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> Lasso:
    """라쏘회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        Lasso: Lasso 모델
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = {"alpha": [0.01, 0.1, 1, 10, 100]}

    return __my_regression(
        classname=Lasso,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_knn_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> KNeighborsRegressor:
    """KNN 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        KNeighborsRegressor
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=KNeighborsRegressor)
    return __my_regression(
        classname=KNeighborsRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_dtree_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    pruning: bool = False,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> DecisionTreeRegressor:
    """DecisionTree 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        DecisionTreeRegressor
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=DecisionTreeRegressor)

        if pruning:
            print("\033[91m가지치기를 위한 alpha값을 탐색합니다.\033[0m")

            try:
                dtree = get_estimator(classname=DecisionTreeRegressor)
                path = dtree.cost_complexity_pruning_path(x_train, y_train)
                ccp_alphas = path.ccp_alphas[1:-1]
                params["ccp_alpha"] = ccp_alphas
            except Exception as e:
                print(f"\033[91m가지치기 실패 ({e})\033[0m")
                e.with_traceback()
        else:
            if "ccp_alpha" in params:
                del params["ccp_alpha"]

            print("\033[91m가지치기를 하지 않습니다.\033[0m")

    return __my_regression(
        classname=DecisionTreeRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        pruning=pruning,
        **params,
    )

@register_method
def my_svr_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> SVR:
    """Support Vector Machine 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        SVR
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=SVR)
    return __my_regression(
        classname=SVR,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_sgd_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> SGDRegressor:
    """SGD 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        SGDRegressor
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=SGDRegressor)

    return __my_regression(
        classname=SGDRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_rf_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> RandomForestRegressor:
    """RandomForest 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        SGDRegressor
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=RandomForestRegressor)

    return __my_regression(
        classname=RandomForestRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = True,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    algorithm: list = None,
    scoring: list = ["rmse", "mse", "r2", "mae", "mape", "mpe", "rf"],
    **params,
) -> any:
    """회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm (list, optional): 사용할 알고리즘 ["linear", "ridge", "lasso", "knn", "dtree", "svr", "sgd"]. Defaults to None.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        any
    """

    results = []  # 결과값을 저장할 리스트
    processes = []  # 병렬처리를 위한 프로세스 리스트
    estimators = {}  # 회귀분석 모델을 저장할 딕셔너리
    estimator_names = []  # 회귀분석 모델의 이름을 저장할 문자열 리스트
    callstack = []
    result_scores = []

    if not algorithm:
        # algorithm = ["linear", "ridge", "lasso", "knn", "dtree", "svr", "sgd", "rf"]
        algorithm = ["linear", "ridge", "lasso", "knn", "dtree", "svr", "sgd"]

    if "linear" in algorithm:
        callstack.append(my_linear_regression)

    if "ridge" in algorithm:
        callstack.append(my_ridge_regression)

    if "lasso" in algorithm:
        callstack.append(my_lasso_regression)

    if "knn" in algorithm:
        callstack.append(my_knn_regression)

    if "dtree" in algorithm:
        callstack.append(my_dtree_regression)

    if "svr" in algorithm:
        callstack.append(my_svr_regression)

    if "sgd" in algorithm:
        callstack.append(my_sgd_regression)

    if "rf" in algorithm:
        callstack.append(my_rf_regression)

    score_fields = []
    score_method = []

    for s in scoring:
        if s == "r2":
            score_fields.append("결정계수(R2)")
            score_method.append(True)
        elif s == "rmse":
            score_fields.append("평균오차(RMSE)")
            score_method.append(False)
        elif s == "mae":
            score_fields.append("평균절대오차(MAE)")
            score_method.append(False)
        elif s == "mse":
            score_fields.append("평균제곱오차(MSE)")
            score_method.append(False)
        elif s == "mape":
            score_fields.append("평균 절대 백분오차 비율(MAPE)")
            score_method.append(False)
        elif s == "mpe":
            score_fields.append("평균 비율 오차(MPE)")
            score_method.append(False)

    # 병렬처리를 위한 프로세스 생성 -> 회귀 모델을 생성하는 함수를 각각 호출한다.
    with futures.ThreadPoolExecutor() as executor:
        for c in callstack:
            if params:
                p = params.copy()

                if c != my_dtree_regression:
                    del p["pruning"]

            else:
                p = {}

            processes.append(
                executor.submit(
                    c,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    cv=cv,
                    learning_curve=False,
                    report=False,
                    plot=False,
                    deg=1,
                    resid_test=False,
                    figsize=figsize,
                    dpi=dpi,
                    sort=False,
                    is_print=False,
                    **p,
                )
            )

        # 병렬처리 결과를 기다린다.
        for p in futures.as_completed(processes):
            # 각 회귀 함수의 결과값(회귀모형 객체)을 저장한다.
            estimator = p.result()

            if estimator is None:
                continue

            # 회귀모형 객체가 포함하고 있는 성능 평가지표(딕셔너리)를 복사한다.
            scores = estimator.scores
            # 회귀모형의 이름과 객체를 저장한다.
            n = estimator.__class__.__name__
            estimator_names.append(n)
            estimators[n] = estimator
            # 성능평가 지표 딕셔너리를 리스트에 저장
            results.append(scores)

            result_scores.append(
                {
                    "model": n,
                    "train": estimator.train_score,
                    "test": estimator.test_score,
                }
            )

        # 결과값을 데이터프레임으로 변환
        print("\n\n==================== 모델 성능 비교 ====================")
        result_df = DataFrame(results, index=estimator_names)

        if score_fields:
            result_df.sort_values(score_fields, ascending=score_method, inplace=True)

        my_pretty_table(result_df)

        score_df = DataFrame(data=result_scores, index=estimator_names).sort_values(
            by="test", ascending=False
        )
        score_df = score_df.melt(id_vars="model", var_name="data", value_name="score")
        my_barplot(
            df=score_df,
            yname="model",
            xname="score",
            hue="data",
            figsize=figsize,
            dpi=dpi,
            callback=lambda ax: ax.set_title("모델 성능 비교"),
        )

    # 최고 성능의 모델을 선택
    if score_fields[0] == "결정계수(R2)":
        best_idx = result_df[score_fields[0]].idxmax()
    else:
        best_idx = result_df[score_fields[0]].idxmin()

    estimators["best"] = estimators[best_idx]

    my_regression_result(
        estimator=estimators["best"],
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        learning_curve=learning_curve,
        cv=cv,
        figsize=figsize,
        dpi=dpi,
        is_print=True,
    )

    if report:
        my_regression_report(
            estimator=estimators["best"],
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            sort=sort,
            plot=plot,
            deg=deg,
            figsize=figsize,
            dpi=dpi,
        )

    if resid_test:
        my_resid_test(
            x=x_train,
            y=y_train,
            y_pred=estimators["best"].predict(x_train),
            figsize=figsize,
            dpi=dpi,
        )

    return estimators

@register_method
def my_voting_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    lr: bool = True,
    rg: bool = False,
    ls: bool = False,
    knn: bool = True,
    dtree: bool = False,
    svr: bool = False,
    sgd: bool = False,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = True,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
) -> VotingRegressor:
    """Voting 분류분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        lr (bool, optional): 로지스틱 회귀분석을 사용할지 여부. Defaults to True.
        rg (bool, optional): 릿지 회귀분석을 사용할지 여부. Defaults to False.
        ls (bool, optional): 라쏘 회귀분석을 사용할지 여부. Defaults to False.
        knn (bool, optional): KNN 회귀분석을 사용할지 여부. Defaults to True.
        dtree (bool, optional): 의사결정나무 회귀분석을 사용할지 여부. Defaults to False.
        svr (bool, optional): 서포트벡터 회귀분석을 사용할지 여부. Defaults to False.
        sgd (bool, optional): SGD 회귀분석을 사용할지 여부. Defaults to False.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to True.
        report (bool, optional): 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (str, optional): 정렬 기준. Defaults to None.
    """

    params = {}
    estimators = []

    if lr:
        estimators.append(("lr", get_estimator(classname=LinearRegression)))
        params.update(get_hyper_params(classname=LinearRegression, key="lr"))

    if rg:
        estimators.append(("rg", get_estimator(classname=Ridge)))
        params.update(get_hyper_params(classname=Ridge, key="rg"))

    if ls:
        estimators.append(("ls", get_estimator(classname=Lasso)))
        params.update(get_hyper_params(classname=Lasso, key="ls"))

    if knn:
        estimators.append(("knn", get_estimator(classname=KNeighborsRegressor)))
        params.update(get_hyper_params(classname=KNeighborsRegressor, key="knn"))

    if dtree:
        estimators.append(("dtree", get_estimator(classname=DecisionTreeRegressor)))
        params.update(get_hyper_params(classname=DecisionTreeRegressor, key="dtree"))

    if svr:
        estimators.append(("svr", get_estimator(classname=SVR)))
        params.update(get_hyper_params(classname=SVR, key="svr"))

    if sgd:
        estimators.append(("sgd", get_estimator(classname=SGDRegressor)))
        params.update(get_hyper_params(classname=SGDRegressor, key="sgd"))

    return __my_regression(
        classname=VotingRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        estimators=estimators,
        **params,
    )

@register_method
def my_bagging_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    estimator: type = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = True,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    algorithm: list = None,
    scoring: list = ["rmse", "mse", "r2", "mae", "mape", "mpe"],
    **params,
) -> DataFrame:
    """배깅 앙상블 분류분석을 수행하고 결과를 출력한다.

    Args:
        estimator (type): 기본 분류분석 알고리즘
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm: list = None,
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        DataFrame: 분류분석 결과
    """

    if estimator is None:
        estimator = my_regression(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cv=cv,
            learning_curve=learning_curve,
            report=False,
            plot=False,
            deg=deg,
            resid_test=False,
            figsize=figsize,
            dpi=dpi,
            sort=sort,
            algorithm=algorithm,
            scoring=scoring,
            **params,
        )

        estimator = estimator["best"]

    if type(estimator) is type:
        params = get_hyper_params(classname=estimator, key="estimator")
        estimator = get_estimator(classname=estimator)
    else:
        params = get_hyper_params(classname=estimator.__class__, key="estimator")

    bagging_params = get_hyper_params(classname=BaggingRegressor)
    params.update(bagging_params)

    return __my_regression(
        classname=BaggingRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        base_estimator=estimator,
        **params,
    )

@register_method
def my_ada_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    estimator: type = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = True,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    algorithm: list = None,
    scoring: list = ["rmse", "mse", "r2", "mae", "mape", "mpe"],
    **params,
) -> AdaBoostRegressor:
    """AdaBoost 앙상블 회귀분석을 수행하고 결과를 출력한다.

    Args:
        estimator (type): 기본 회귀분석 알고리즘
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm: list = None,
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        AdaBoostRegressor
    """

    if estimator is None:
        estimator = my_regression(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cv=cv,
            learning_curve=learning_curve,
            report=False,
            plot=False,
            deg=deg,
            resid_test=False,
            figsize=figsize,
            dpi=dpi,
            sort=sort,
            algorithm=algorithm,
            scoring=scoring,
            **params,
        )

        estimator = estimator["best"]

    if type(estimator) is type:
        params = get_hyper_params(classname=estimator, key="estimator")
        estimator = get_estimator(classname=estimator)
    else:
        params = get_hyper_params(classname=estimator.__class__, key="estimator")

    bagging_params = get_hyper_params(classname=AdaBoostRegressor)
    params.update(bagging_params)

    return __my_regression(
        classname=AdaBoostRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        base_estimator=estimator,
        **params,
    )

@register_method
def my_gbm_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = True,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    scoring: list = ["rmse", "mse", "r2", "mae", "mape", "mpe"],
    **params,
) -> GradientBoostingRegressor:
    """GradientBoosting 앙상블 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series): 훈련 데이터의 종속변수
        x_test (DataFrame, optional): 검증 데이터의 독립변수. Defaults to None.
        y_test (Series, optional): 검증 데이터의 종속변수. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        algorithm: list = None,
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        GradientBoostingRegressor
    """

    params = get_hyper_params(classname=AdaBoostRegressor)

    return __my_regression(
        classname=GradientBoostingRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=True,
        **params,
    )


@register_method
def my_xgb_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    pruning: bool = False,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> XGBRegressor:
    """XGBRegressor 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        XGBRegressor
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=XGBRegressor)

    return __my_regression(
        classname=XGBRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )

@register_method
def my_lgbm_regression(
    x_train: DataFrame,
    y_train: Series,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    learning_curve: bool = True,
    report=True,
    plot: bool = False,
    deg: int = 1,
    resid_test=False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> LGBMRegressor:
    """LGBMRegressor 회귀분석을 수행하고 결과를 출력한다.

    Args:
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 0.
        learning_curve (bool, optional): 학습곡선을 출력할지 여부. Defaults to False.
        report (bool, optional): 회귀분석 결과를 보고서로 출력할지 여부. Defaults to True.
        plot (bool, optional): 시각화 여부. Defaults to True.
        deg (int, optional): 다항회귀분석의 차수. Defaults to 1.
        resid_test (bool, optional): 잔차의 가정을 확인할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        sort (bool, optional): 독립변수 결과 보고 표의 정렬 기준 (v, p)
        is_print (bool, optional): 출력 여부. Defaults to True.
        pruning (bool, optional): 의사결정나무에서 가지치기의 alpha값을 하이퍼 파라미터 튜닝에 포함 할지 여부. Default to False.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        LGBMRegressor
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = get_hyper_params(classname=LGBMRegressor)

    return __my_regression(
        classname=LGBMRegressor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv=cv,
        learning_curve=learning_curve,
        report=report,
        plot=plot,
        deg=deg,
        resid_test=resid_test,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )
