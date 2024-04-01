import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import concurrent.futures as futures

from typing import Literal
from pandas import DataFrame
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial import ConvexHull

from .plot import my_lineplot, my_convex_hull

__RANDOM_STATE__ = 0


def __kmeans(
    data: DataFrame,
    n_clusters: int,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=__RANDOM_STATE__,
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
) -> KMeans:
    """KMmeans 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int): 클러스터 개수
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): 알고리즘. Defaults to "lloyd".

    Returns:
        KMeans
    """
    estimator = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm=algorithm,
    )
    estimator.fit(data)

    # 속성 확장
    estimator.n_clusters = n_clusters
    estimator.silhouette = silhouette_score(data, estimator.labels_)
    return estimator


def my_single_kmeans(
    data: DataFrame,
    n_clusters: int,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=__RANDOM_STATE__,
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
) -> KMeans:
    """kmeans 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int): 클러스터 개수
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): 알고리즘. Defaults to "lloyd".

    Returns:
        KMeans
    """
    return __kmeans(
        data=data,
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm=algorithm,
    )


def my_elbow_point(
    x: list,
    y: list,
    dir: Literal["left,down", "left,up", "right,down", "right,up"] = "left,down",
    title: str = None,
    xname: str = None,
    yname: str = None,
    S: float = 0.1,
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> tuple:
    """엘보우 포인트를 찾는다.

    Args:
        x (list): x축 데이터
        y (list): y축 데이터
        dir (str, optional): 그래프가 볼록한 방향. Defaults to "left,down".
        title (str, optional): 그래프 제목. Defaults to "Elbow Method".
        xname (str, optional): x축 이름. Defaults to "n_clusters".
        yname (str, optional): y축 이름. Defaults to "inertia".
        S (float, optional): 정밀도. Defaults to 1.0.
        plot (bool, optional): 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.

    Returns:
        int: _description_
    """
    # left,down  convex, decresing
    # left,up concave, increasing
    # right,down convex, increasing
    # right,up  concave, decresing

    if dir == "left,down":
        curve = "convex"
        direction = "decreasing"
    elif dir == "left,up":
        curve = "concave"
        direction = "increasing"
    elif dir == "right,down":
        curve = "convex"
        direction = "increasing"
    else:
        curve = "concave"
        direction = "decreasing"

    kn = KneeLocator(x=x, y=y, curve=curve, direction=direction, S=S)

    best_x = kn.elbow
    best_y = kn.elbow_y

    if plot:

        def hvline(ax):
            if xname:
                ax.set_xlabel(xname)

            if yname:
                ax.set_ylabel(yname)

            if title:
                ax.set_title(title)

            ax.axvline(best_x, color="red", linestyle="--", linewidth=0.7)
            ax.axhline(best_y, color="red", linestyle="--", linewidth=0.7)
            ax.text(
                best_x + 0.2,
                best_y + 0.2,
                "(%.1f, %.1f)" % (best_x, best_y),
                fontsize=20,
                color="red",
                va="bottom",
                ha="left",
            )

        my_lineplot(
            df=None,
            xname=x,
            yname=y,
            marker="o",
            linewidth=2,
            figsize=figsize,
            dpi=dpi,
            callback=hvline,
        )

    return (best_x, best_y)


def __silhouette_plot(cluster: any, data: DataFrame, ax: plt.Axes) -> None:
    """실루엣 계수를 파라미터로 전달받은 ax에 시각화 한다.

    Args:
        clusters (list): 클러스터 개수 리스트
        data (DataFrame): 원본 데이터
        ax (plt.Axes): 그래프 객체
    """
    sil_avg = silhouette_score(X=data, labels=cluster.labels_)
    sil_values = silhouette_samples(X=data, labels=cluster.labels_)

    y_lower = 10
    ax.set_title(
        "Number of Cluster : " + str(cluster.n_clusters) + ", "
        "Silhouette Score :" + str(round(sil_avg, 3))
    )
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(data) + (cluster.n_clusters + 1) * 10])
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.grid()

    # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
    for i in range(cluster.n_clusters):
        ith_cluster_sil_values = sil_values[cluster.labels_ == i]
        ith_cluster_sil_values.sort()

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--")


def my_cluster_plot(
    estimator: any, data: DataFrame, figsize: tuple = (10, 5), dpi: int = 100
) -> None:
    """클러스터링 결과를 시각화한다.

    Args:
        estimator (any): 클러트링 객체
        data (DataFrame): 원본 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.
    """
    df = data.copy()
    df["cluster"] = estimator.labels_

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(figsize[0] * 2, figsize[1]), dpi=dpi
    )
    fig.subplots_adjust(wspace=0.1)

    # 1st Plot showing the silhouette plot
    __silhouette_plot(cluster=estimator, data=data, ax=ax1)

    # 2nd Plot showing the actual clusters formed
    xname = data.columns[0]
    yname = data.columns[1]

    for c in df["cluster"].unique():
        if c == -1:
            continue

        # 한 종류만 필터링한 결과에서 두 변수만 선택
        df_c = df.loc[df["cluster"] == c, [xname, yname]]

        try:
            # 외각선 좌표 계산
            hull = ConvexHull(df_c)

            # 마지막 좌표 이후에 첫 번째 좌표를 연결
            points = np.append(hull.vertices, hull.vertices[0])

            ax2.plot(
                df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":"
            )
            ax2.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1)
        except:
            pass

    sb.scatterplot(data=df, x=xname, y=yname, hue="cluster", ax=ax2)

    # Labeling the clusters
    centers = estimator.cluster_centers_
    # Draw white circles at cluster centers
    sb.scatterplot(
        x=centers[:, 0],
        y=centers[:, 1],
        marker="o",
        color="white",
        alpha=1,
        s=200,
        edgecolor="r",
        ax=ax2,
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    ax2.grid()

    plt.show()
    plt.close()


def my_silhouette_plot(
    clusters: list,
    data: DataFrame,
    figsize: tuple = (8, 6),
    dpi: int = 100,
    cols: int = 3,
) -> None:
    """실루엣 계수를 시각화한다.

    Args:
        clusters (list): 클러스터 개수 리스트
        data (DataFrame): 원본 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.
        cols (int, optional): 열 개수. Defaults to 3.
    """
    rows = (len(clusters) + cols - 1) // cols
    fig, ax = plt.subplots(
        nrows=rows, ncols=cols, figsize=(figsize[0] * rows, figsize[1] * cols), dpi=dpi
    )

    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    with futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(clusters)):
            cl = clusters[i]
            r = i // cols
            c = i % cols
            executor.submit(__silhouette_plot, cluster=cl, data=data, ax=ax[r, c])

    plt.show()
    plt.close()


def my_kmeans(
    data: DataFrame,
    n_clusters: int | list = 10,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=__RANDOM_STATE__,
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    scoring: Literal["elbow", "e", "silhouette", "s"] = "elbow",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DataFrame:
    """클러스터 개수에 따른 이너셔 값을 계산한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int | list, optional): 최대 클러스터 개수. 정수로 전달할 경우 `2`부터 주어진 개수까지 반복 수행한다. Defaults to 10.
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): _description_. Defaults to "lloyd".

    Returns:
        DataFrame: _description_
    """
    if isinstance(n_clusters, int):
        n_clusters = list(range(2, n_clusters + 1))

    with futures.ThreadPoolExecutor() as executor:
        results = []
        for n in n_clusters:
            results.append(
                executor.submit(
                    __kmeans,
                    data,
                    n_clusters=n,
                    init=init,
                    max_iter=max_iter,
                    random_state=random_state,
                    algorithm=algorithm,
                )
            )

        # 비동기처리로 생성된 군집객체들을 수집 --> 비동기이므로 먼저 종료된 순서대로 수집된다.
        kmeans_list = [r.result() for r in futures.as_completed(results)]

        # 클러스터 개수로 정렬
        kmeans_list = sorted(kmeans_list, key=lambda x: x.n_clusters)

        # 클러스터 개수만 별도로 추출
        cluster_list = [k.n_clusters for k in kmeans_list]

        # 최적 모델을 저장할 객체
        best_model = None

        if scoring == "elbow" or scoring == "e":
            inertia = [k.inertia_ for k in kmeans_list]

            best_k, best_y = my_elbow_point(
                x=cluster_list,
                y=inertia,
                dir="left,down",
                xname="n_cluster",
                yname="inertia",
                title="Elbow Method",
                plot=plot,
                figsize=figsize,
                dpi=dpi,
            )

            best_model = next(filter(lambda x: x.n_clusters == best_k, kmeans_list))
        elif scoring == "silhouette" or scoring == "s":
            best_model = max(kmeans_list, key=lambda x: x.silhouette)

            if plot:
                my_silhouette_plot(kmeans_list, data, figsize=(8, 6), dpi=dpi)

        # 최종 군집 결과를 시각화 한다.
        if plot:
            my_cluster_plot(best_model, data, figsize=figsize, dpi=dpi)

        return best_model