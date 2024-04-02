import inspect

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

# from .util import *

__RANDOM_STATE__ = 0

__MAX_ITER__ = 1000

__N_JOBS__ = -1

__LOGISTIC_REGRESSION_HYPER_PARAMS__ = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
}

__KNN_CLASSFICATION_HYPER_PARAMS__ = {
    "n_neighbors": np.arange(2, stop=10),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}

__NB_HYPER_PARAMS__ = {
    # "priors" : None,
    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

__DTREE_HYPER_PARAMS__ = {
    "criterion": ["gini", "entropy"],
    "max_depth": np.arange(1, stop=10),
    "min_samples_split": np.arange(2, stop=10),
    "min_samples_leaf": np.arange(1, stop=10),
    "max_features": ["auto", "sqrt", "log2"],
    "ccp_alpha": [0.0],
}

__LINEAR_SVC_HYPER_PARAMS__ = {
    "penalty": ["l1", "l2"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
}

__SVC_HYPER_PARAMS__ = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "kernel": ["poly", "rbf", "sigmoid"],
    "degree": np.arange(2, stop=10),
    "gamma": ["scale", "auto"],
}

__SGD_HYPER_PARAMS__ = {
    "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
}


def __ml(
    classname: any,
    x_train: DataFrame,
    y_train: Series = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    scoring: any = None,
    is_print: bool = True,
    pruning: bool = False,
    **params,
) -> any:
    """머신러닝 분석을 수행하고 결과를 출력한다.

    Args:
        classname (any): 분류분석 추정기 (모델 객체)
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        is_print (bool, optional): 출력 여부. Defaults to True.

    Returns:
        any: 분류분석 모델
    """

    # 교차검증 설정
    if cv > 0:
        if not params:
            params = {}

        args = {}

        c = str(classname)
        p = c.rfind(".")
        cn = c[p + 1 : -2]

        if "n_jobs" in dict(inspect.signature(classname.__init__).parameters):
            args["n_jobs"] = __N_JOBS__

        if "max_iter" in dict(inspect.signature(classname.__init__).parameters):
            args["max_iter"] = __MAX_ITER__

        if "random_state" in dict(inspect.signature(classname.__init__).parameters):
            args["random_state"] = __RANDOM_STATE__

        if "early_stopping" in dict(inspect.signature(classname.__init__).parameters):
            args["early_stopping"] = True

        if "probability" in dict(inspect.signature(classname.__init__).parameters):
            args["probability"] = True

        prototype_estimator = classname(**args)
        print(f"\033[92m{cn}({args}) {params}\033[0m".replace("\n", ""))

        if pruning and (
            classname == DecisionTreeClassifier or classname == DecisionTreeRegressor
        ):
            try:
                dtree = classname(**args)
                path = dtree.cost_complexity_pruning_path(x_train, y_train)
                ccp_alphas = path.ccp_alphas[1:-1]
                params["ccp_alpha"] = ccp_alphas
            except Exception as e:
                print(f"\033[91m{cn}의 가지치기 실패 ({e})\033[0m")

        # grid = GridSearchCV(
        #     prototype_estimator, param_grid=params, cv=cv, n_jobs=-1
        # )
        if scoring is None:
            grid = RandomizedSearchCV(
                estimator=prototype_estimator,
                param_distributions=params,
                cv=cv,
                n_jobs=__N_JOBS__,
                n_iter=__MAX_ITER__,
            )
        else:
            grid = RandomizedSearchCV(
                estimator=prototype_estimator,
                param_distributions=params,
                cv=cv,
                n_jobs=__N_JOBS__,
                n_iter=__MAX_ITER__,
                scoring=scoring,
            )

        try:
            grid.fit(x_train, y_train)
        except Exception as e:
            print(f"\033[91m{cn}에서 에러발생 ({e})\033[0m")
            return None

        # print(grid.cv_results_)

        result_df = DataFrame(grid.cv_results_["params"])
        # result_df["mean_test_score"] = grid.cv_results_["mean_test_score"]

        estimator = grid.best_estimator_
        estimator.best_params = grid.best_params_

        if is_print:
            # print("[교차검증 TOP5]")
            # my_pretty_table(
            #     result_df.dropna(subset=["mean_test_score"])
            #     .sort_values(by="mean_test_score", ascending=False)
            #     .head()
            # )
            # my_pretty_table(result_df)
            print("")

            print("[Best Params]")
            print(grid.best_params_)
            print("")
    else:
        if "n_jobs" in dict(inspect.signature(classname.__init__).parameters):
            params["n_jobs"] = __N_JOBS__
        else:
            print("%s는 n_jobs를 허용하지 않음" % classname)

        if "random_state" in dict(inspect.signature(classname.__init__).parameters):
            params["random_state"] = __RANDOM_STATE__
        else:
            print("%s는 random_state를 허용하지 않음" % classname)

        estimator = classname(**params)
        estimator.fit(x_train, y_train)

    # ------------------------------------------------------
    # 결과값 생성

    # 훈련 데이터에 대한 추정치 생성
    y_pred = (
        estimator.predict(x_test) if x_test is not None else estimator.predict(x_train)
    )

    if hasattr(estimator, "predict_proba"):
        y_pred_prob = (
            estimator.predict_proba(x_test)
            if x_test is not None
            else estimator.predict_proba(x_train)
        )

    # 도출된 결과를 모델 객체에 포함시킴
    estimator.x = x_test if x_test is not None else x_train
    estimator.y = y_test if y_test is not None else y_train
    estimator.y_pred = y_pred if y_test is not None else estimator.predict(x_train)

    if y_test is not None or y_train is not None:
        estimator.resid = (
            y_test - y_pred
            if y_test is not None
            else y_train - estimator.predict(x_train)
        )

    if hasattr(estimator, "predict_proba"):
        estimator.y_pred_proba = (
            y_pred_prob if y_test is not None else estimator.predict_proba(x_train)
        )

    return estimator


def get_random_state() -> int:
    """랜덤 시드를 반환한다.

    Returns:
        int: 랜덤 시드
    """
    return __RANDOM_STATE__


def get_hyper_params(classname: any) -> dict:
    """분류분석 추정기의 하이퍼파라미터를 반환한다.

    Args:
        classname (any): 분류분석 추정기

    Returns:
        dict: 하이퍼파라미터
    """

    if classname == LogisticRegression:
        return __LOGISTIC_REGRESSION_HYPER_PARAMS__
    elif classname == KNeighborsClassifier:
        return __KNN_CLASSFICATION_HYPER_PARAMS__
    elif classname == GaussianNB:
        return __NB_HYPER_PARAMS__
    elif classname == DecisionTreeClassifier:
        return __DTREE_HYPER_PARAMS__
    elif classname == LinearSVC:
        return __LINEAR_SVC_HYPER_PARAMS__
    elif classname == SVC:
        return __SVC_HYPER_PARAMS__
    elif classname == SGDClassifier:
        return __SGD_HYPER_PARAMS__
    else:
        return {}
