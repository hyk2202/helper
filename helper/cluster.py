import inspect

import numpy as np

from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, make_scorer
from .util import my_pretty_table
from .core import __ml


def __my_clustering(
    classname: any,
    x: DataFrame,
    cv: int = 5,
    report: bool = False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> any:

    # ------------------------------------------------------
    # 분류모델 생성
    estimator = __ml(
        classname=classname,
        x_train=x,
        cv=cv,
        scoring=make_scorer(score_func=silhouette_score),
        is_print=is_print,
        **params,
    )

    # ------------------------------------------------------
    # 성능평가

    # ------------------------------------------------------
    # 보고서 출력

    return estimator


def my_kmeans(
    x: DataFrame,
    n_clusters: any = [2, 3, 4, 5, 6, 7, 8, 9, 10],
    cv: int = 5,
    report: bool = False,
    figsize=(10, 5),
    dpi: int = 100,
    sort: str = None,
    is_print: bool = True,
    **params,
) -> KMeans:
    """KMeans 클러스터링을 수행하고 결과를 출력한다.

    Args:
        x (DataFrame): 독립변수에 대한 데이터
        n_clusters (int, optional): 클러스터의 개수. Defaults to 8.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        report (bool, optional): 보고서 출력 여부. Defaults to False.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
        sort (str, optional): 정렬 기준. Defaults to None.
        is_print (bool, optional): 출력 여부. Defaults to True.

    Returns:
        KMeans: 클러스터링 모델
    """
    if type(n_clusters) is not list:
        n_clusters = [n_clusters]

    if params:
        params["n_clusters"] = n_clusters
    else:
        params = {
            "n_clusters": n_clusters,
            "algorithm": ["lloyd", "elkan"],
        }

    return __my_clustering(
        classname=KMeans,
        x=x,
        cv=cv,
        eport=report,
        figsize=figsize,
        dpi=dpi,
        sort=sort,
        is_print=is_print,
        **params,
    )
