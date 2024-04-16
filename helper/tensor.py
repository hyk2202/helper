# -*- coding: utf-8 -*-
# -------------------------------------------------------------

import sys
import numpy as np
from datetime import datetime as dt

# -------------------------------------------------------------
from pycallgraphix.wrapper import register_method

# -------------------------------------------------------------
from pandas import DataFrame

# -------------------------------------------------------------
from matplotlib import pyplot as plt

# -------------------------------------------------------------
from tensorflow.random import set_seed
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import (
    History,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam, RMSprop

# -------------------------------------------------------------
from kerastuner import Hyperband

# -------------------------------------------------------------
from .util import my_pretty_table
from .core import get_random_state
from .regression import my_regression_result, my_regression_report
from .classification import my_classification_result, my_classification_report

# -------------------------------------------------------------

set_seed(get_random_state())
__initializer__ = GlorotUniform(seed=get_random_state())

if sys.platform == "darwin":
    __HB_DIR__ = "tf_hyperband"
else:
    __HB_DIR__ = "D:\\tf_hyperband"


def __get_project_name(src) -> str:
    if src:
        return src

    return "tf_%s" % dt.now().strftime("%y%m%d_%H%M%S")


# -------------------------------------------------------------
@register_method
def tf_tune(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense_tune: list = [],
    optimizer: any = "adam",
    learning_rate: list = [1e-2, 1e-3, 1e-4],
    loss: str = None,
    metrics: list = None,
    epochs: int = 500,
    batch_size: int = 32,
    factor: int = 3,
    seed: int = get_random_state(),
    directory: str = __HB_DIR__,
    project_name: str = None,
) -> Sequential:
    """_summary_

    Args:
        x_train (np.ndarray): _description_
        y_train (np.ndarray): _description_
        x_test (np.ndarray, optional): _description_. Defaults to None.
        y_test (np.ndarray, optional): _description_. Defaults to None.
        dense (list, optional): _description_. Defaults to [].
        optimizer (any, optional): _description_. Defaults to "adam".
        learning_rate (list, optional): _description_. Defaults to [1e-2, 1e-3, 1e-4].
        loss (str, optional): _description_. Defaults to None.
        metrics (list, optional): _description_. Defaults to None.
        epochs (int, optional): _description_. Defaults to 10.
        factor (int, optional): _description_. Defaults to 3.
        seed (_type_, optional): _description_. Defaults to get_random_state().
        directory (str, optional): _description_. Defaults to "./tensor_hyperband".
        project_name (_type_, optional): _description_. Defaults to "tf_hyperband_%s"%dt.now().strftime("%Y%m%d%H%M%S").

    Returns:
        Sequential: _description_
    """

    def __tf_build(hp) -> Sequential:
        model = Sequential()

        for d in dense_tune:
            if "input_shape" in d:
                model.add(
                    Dense(
                        units=(
                            hp.Choice("units", values=d["units"])
                            if type(d["units"]) == list
                            else d["units"]
                        ),
                        input_shape=d["input_shape"],
                        activation=d["activation"],
                    )
                )
            else:
                model.add(
                    Dense(
                        units=(
                            hp.Choice("units", values=d["units"])
                            if type(d["units"]) == list
                            else d["units"]
                        ),
                        activation=d["activation"],
                    )
                )

        opt = None

        if optimizer == "adam":
            opt = Adam(hp.Choice("learning_rate", values=learning_rate))
        elif optimizer == "rmsprop":
            opt = RMSprop(hp.Choice("learning_rate", values=learning_rate))

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics,
        )

        return model

    tuner = Hyperband(
        hypermodel=__tf_build,
        objective=f"val_{metrics[0]}",
        max_epochs=epochs,
        factor=factor,
        seed=seed,
        directory=directory,
        project_name=__get_project_name(project_name),
    )

    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()

    if not best_hps:
        raise ValueError("No best hyperparameters found.")

    model = tuner.hypermodel.build(best_hps[0])
    return model


# -------------------------------------------------------------
@register_method
def tf_create(
    dense: list = [],
    optimizer: any = "adam",
    loss: str = None,
    metrics: list = None,
    model_path: str = None,
) -> Sequential:
    """
    지정된 밀집 레이어, 최적화 프로그램, 손실 함수 및 측정항목을 사용하여 TensorFlow Sequential 모델을 생성하고 컴파일한다.

    Args:
        dense (list, optional): 각 사전이 생성될 신경망 모델의 레이어를 나타내는 사전 목록. Defaults to [].
        optimizer (any, optional): 훈련 중에 사용할 최적화 알고리즘. Defaults to "adam".
        loss (str, optional): 신경망 모델 학습 중에 최적화할 손실 함수를 지정. Defaults to None.
        metrics (list, optional): 모델 학습 중에 모니터링하려는 평가 측정항목. Defaults to None.
        model_path (str, optional): 로드하고 반환하려는 저장된 모델의 경로. Defaults to None.

    Raises:
        ValueError: dense, loss 및 metrics는 필수 인수

    Returns:
        Sequential: 컴파일 된 TensorFlow Sequential 모델
    """

    if model_path:
        return load_model(model_path)

    if not dense or not loss or not metrics:
        raise ValueError("dense, loss, and metrics are required arguments")

    model = Sequential()

    for i, v in enumerate(dense):
        if "input_shape" in v:
            model.add(
                Dense(
                    units=v["units"],
                    input_shape=v["input_shape"],
                    activation=v["activation"],
                    kernel_initializer=__initializer__,
                )
            )
        else:
            model.add(
                Dense(
                    units=v["units"],
                    activation=v["activation"],
                    kernel_initializer=__initializer__,
                )
            )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# -------------------------------------------------------------
@register_method
def tf_train(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    checkpoint_path: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
) -> History:
    """파라미터로 전달된 tensroflow 모델을 사용하여 지정된 데이터로 훈련을 수행하고 결과를 반환한다.

    Args:
        model (Sequential): 컴파일된 tensroflow 모델
        x_train (np.ndarray): 훈련 데이터에 대한 독립변수
        y_train (np.ndarray): 훈련 데이터에 대한 종속변수
        x_test (np.ndarray, optional): 테스트 데이터에 대한 독립변수. Defaults to None.
        y_test (np.ndarray, optional): 테스트 데이터에 대한 종속변수. Defaults to None.
        epochs (int, optional): epoch 수. Defaults to 500.
        batch_size (int, optional): 배치 크기. Defaults to 32.
        early_stopping (bool, optional): 학습 조기 종료 기능 활성화 여부. Defaults to True.
        reduce_lr (bool, optional): 학습률 감소 기능 활성화 여부. Defaults to True.
        checkpoint_path (str, optional): 체크포인트가 저장될 파일 경로. Defaults to None.
        tensorboard_path (str, optional): 텐서보드 로그가 저장될 디렉토리 경로. Defaults to None.
        verbose (int, optional): 학습 과정 출력 레벨. Defaults to 0.

    Returns:
        History: 훈련 결과
    """

    callbacks = []

    if early_stopping:
        callbacks.append(
            EarlyStopping(patience=10, restore_best_weights=True, verbose=verbose)
        )

    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(factor=0.1, patience=5, verbose=verbose))

    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose,
            )
        )

    if tensorboard_path:
        callbacks.append(
            TensorBoard(log_dir=tensorboard_path, histogram_freq=1, write_graph=True)
        )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test) if x_test is not None else None,
        verbose=verbose,
        callbacks=callbacks,
    )

    dataset = []
    result_set = []

    if x_train is not None and y_train is not None:
        dataset.append("train")
        result_set.append(model.evaluate(x_train, y_train, verbose=0, return_dict=True))

    if x_test is not None and y_test is not None:
        dataset.append("test")
        result_set.append(model.evaluate(x_test, y_test, verbose=0, return_dict=True))

    result_df = DataFrame(result_set, index=dataset)
    my_pretty_table(result_df)

    return history


# -------------------------------------------------------------
@register_method
def tf_result(
    result: History,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
) -> Sequential:
    """훈련 결과를 시각화하고 표로 출력한다.

    Args:
        result (History): 훈련 결과
        history_table (bool, optional): 훈련 결과를 표로 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프 크기. Defaults to (7, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.
    Returns:
        Sequential: 훈련된 TensorFlow Sequential 모델
    """
    result_df = DataFrame(result.history)
    result_df["epochs"] = result_df.index + 1
    result_df.set_index("epochs", inplace=True)

    columns = result_df.columns[:-1]
    s = len(columns)

    group_names = []

    for i in range(0, s - 1):
        if columns[i][:3] == "val":
            break

        t = f"val_{columns[i]}"
        c2 = list(columns[i + 1 :])

        try:
            var_index = c2.index(t)
        except:
            var_index = -1

        if var_index > -1:
            group_names.append([columns[i], t])
        else:
            group_names.append([columns[i]])

    cols = len(group_names)

    fig, ax = plt.subplots(1, cols, figsize=(figsize[0] * cols, figsize[1]), dpi=dpi)

    if cols == 1:
        ax = [ax]

    for i in range(0, cols):
        result_df.plot(y=group_names[i], ax=ax[i])
        ax[i].grid()

    plt.show()
    plt.close()

    if history_table:
        my_pretty_table(result_df)


# -------------------------------------------------------------
@register_method
def my_tf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense: list = [],
    optimizer: any = "adam",
    loss: str = None,
    metrics: list = None,
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    checkpoint_path: str = None,
    model_path: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
    # hyperband parameters
    dense_tune: list = [],
    learning_rate: list = [1e-2, 1e-3, 1e-4],
    factor=3,
    seed=get_random_state(),
    directory=__HB_DIR__,
    project_name=None,
) -> Sequential:
    """
    텐서플로우 학습 모델을 생성하고 훈련한 후 결과를 출력한다.

    Args:
        x_train (np.ndarray): 훈련 데이터에 대한 독립변수
        y_train (np.ndarray): 훈련 데이터에 대한 종속변수
        x_test (np.ndarray, optional): 테스트 데이터에 대한 독립변수. Defaults to None.
        y_test (np.ndarray, optional): 테스트 데이터에 대한 종속변수. Defaults to None.
        dense (list, optional): 각 사전이 생성될 신경망 모델의 레이어를 나타내는 사전 목록. Defaults to [].
        optimizer (any, optional): 훈련 중에 사용할 최적화 알고리즘. Defaults to "adam".
        loss (str, optional): 신경망 모델 학습 중에 최적화할 손실 함수를 지정. Defaults to None.
        metrics (list, optional): 모델 학습 중에 모니터링하려는 평가 측정항목. Defaults to None.
        epochs (int, optional): epoch 수. Defaults to 500.
        batch_size (int, optional): 배치 크기. Defaults to 32.
        early_stopping (bool, optional): 학습 조기 종료 기능 활성화 여부. Defaults to True.
        reduce_lr (bool, optional): 학습률 감소 기능 활성화 여부. Defaults to True.
        checkpoint_path (str, optional): 체크포인트가 저장될 파일 경로. Defaults to None.
        model_path (str, optional): _description_. Defaults to None.
        tensorboard_path (str, optional): 텐서보드 로그가 저장될 디렉토리 경로. Defaults to None.
        verbose (int, optional): 학습 과정 출력 레벨. Defaults to 0.
        history_table (bool, optional): 훈련 결과를 표로 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프 크기. Defaults to (7, 5).
        dpi (int, optional): 그래프 해상도. Defaults to 100.

    Returns:
        Sequential: 훈련된 TensorFlow Sequential 모델
    """
    if dense_tune is not None and len(dense_tune) > 0:
        model = tf_tune(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dense_tune=dense_tune,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            factor=factor,
            seed=seed,
            directory=directory,
            project_name=__get_project_name(project_name),
        )
    else:
        model = tf_create(
            dense=dense,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            model_path=model_path,
        )

    result = tf_train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        reduce_lr=reduce_lr,
        checkpoint_path=checkpoint_path,
        tensorboard_path=tensorboard_path,
        verbose=verbose,
    )

    tf_result(result=result, history_table=history_table, figsize=figsize, dpi=dpi)

    return model


# -------------------------------------------------------------
@register_method
def my_tf_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense_units: list = [64, 32],
    optimizer: any = "adam",
    loss: any = "mse",
    metrics=["mae"],
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    model_path: str = None,
    checkpoint_path: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
    history_table: bool = False,
    deg: int = 1,
    figsize: tuple = (7, 5),
    dpi: int = 100,
    # hyperband parameters
    dense_tune: list = [
        {"units": [128, 64, 32, 16, 8], "activation": "relu", "input_shape": (0,)},
        {"units": [64, 32, 16, 8, 4], "activation": "relu"},
        {"units": 1, "activation": "linear"},
    ],
    learning_rate: list = [1e-2, 1e-3, 1e-4],
    factor=3,
    seed=get_random_state(),
    directory=__HB_DIR__,
    project_name=None,
    plot: bool = True,
) -> Sequential:
    if dense_tune is not None and len(dense_tune) > 0:
        dense_tune[0]["input_shape"] = (x_train.shape[1],)

        model = my_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            tensorboard_path=tensorboard_path,
            verbose=verbose,
            history_table=history_table,
            figsize=figsize,
            dpi=dpi,
            # hyperband parameters
            dense_tune=dense_tune,
            learning_rate=learning_rate,
            factor=factor,
            seed=seed,
            directory=directory,
            project_name=__get_project_name(project_name),
        )
    else:
        dense = []

        s = len(dense_units)
        for i, v in enumerate(iterable=dense_units):
            if i == 0:
                dense.append(
                    {
                        "units": v,
                        "input_shape": (x_train.shape[1],),
                        "activation": "relu",
                    }
                )
            else:
                dense.append({"units": v, "activation": "relu"})

        dense.append({"units": 1, "activation": "linear"})

        model = my_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dense=dense,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            reduce_lr=reduce_lr,
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            tensorboard_path=tensorboard_path,
            verbose=verbose,
            history_table=history_table,
            figsize=figsize,
            dpi=dpi,
        )

    # my_regression_result(
    #     estimator=model,
    #     x_train=x_train,
    #     y_train=y_train,
    #     x_test=x_test,
    #     y_test=y_test,
    #     figsize=figsize,
    #     dpi=dpi,
    # )

    # my_regression_report(
    #     estimator=model,
    #     x_train=x_train,
    #     y_train=y_train,
    #     x_test=x_test,
    #     y_test=y_test,
    #     plot=plot,
    #     figsize=figsize,
    #     dpi=dpi,
    #     deg=deg,
    # )

    return model


# -------------------------------------------------------------
@register_method
def my_tf_sigmoid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense_units: list = [64, 32],
    optimizer: any = "rmsprop",
    loss: any = "binary_crossentropy",
    metrics=["acc"],
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    model_path: str = None,
    checkpoint_path: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
    # hyperband parameters
    dense_tune: list = [
        {"units": [256, 128, 64, 32], "activation": "relu", "input_shape": (0,)},
        {"units": 1, "activation": "sigmoid"},
    ],
    learning_rate: list = [1e-2, 1e-3, 1e-4],
    factor=3,
    seed=get_random_state(),
    directory=__HB_DIR__,
    project_name=None,
) -> Sequential:
    if dense_tune is not None and len(dense_tune) > 0:
        dense_tune[0]["input_shape"] = (x_train.shape[1],)

        model = my_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            tensorboard_path=tensorboard_path,
            verbose=verbose,
            history_table=history_table,
            figsize=figsize,
            dpi=dpi,
            # hyperband parameters
            dense_tune=dense_tune,
            learning_rate=learning_rate,
            factor=factor,
            seed=seed,
            directory=directory,
            project_name=__get_project_name(project_name),
        )
    else:
        dense = []

        s = len(dense_units)
        for i, v in enumerate(iterable=dense_units):
            if i == 0:
                dense.append(
                    {
                        "units": v,
                        "input_shape": (x_train.shape[1],),
                        "activation": "relu",
                    }
                )
            else:
                dense.append({"units": v, "activation": "relu"})

        dense.append({"units": 1, "activation": "sigmoid"})

        model = my_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dense=dense,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            reduce_lr=reduce_lr,
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            tensorboard_path=tensorboard_path,
            verbose=verbose,
            history_table=history_table,
            figsize=figsize,
            dpi=dpi,
        )

    # my_classification_result(
    #     estimator=model,
    #     x_train=x_train,
    #     y_train=y_train,
    #     x_test=x_test,
    #     y_test=y_test,
    #     figsize=figsize,
    #     dpi=dpi,
    # )

    # my_classification_report(
    #     estimator=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    # )

    return model


# -------------------------------------------------------------
@register_method
def my_tf_softmax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense_units: list = [64, 32],
    optimizer: any = "rmsprop",
    loss: any = "categorical_crossentropy",
    metrics=["acc"],
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    model_path: str = None,
    checkpoint_path: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
    # hyperband parameters
    dense_tune: list = [
        {"units": [256, 128, 64, 32], "activation": "relu", "input_shape": (0,)},
        {"units": 1, "activation": "softmax"},
    ],
    learning_rate: list = [1e-2, 1e-3, 1e-4],
    factor=3,
    seed=get_random_state(),
    directory=__HB_DIR__,
    project_name=None,
) -> Sequential:
    if dense_tune is not None and len(dense_tune) > 0:
        dense_tune[0]["input_shape"] = (x_train.shape[1],)
        dense_tune[-1]["units"] = len(y_train[0])

        model = my_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            tensorboard_path=tensorboard_path,
            verbose=verbose,
            history_table=history_table,
            figsize=figsize,
            dpi=dpi,
            # hyperband parameters
            dense_tune=dense_tune,
            learning_rate=learning_rate,
            factor=factor,
            seed=seed,
            directory=directory,
            project_name=__get_project_name(project_name),
        )
    else:
        dense = []

        s = len(dense_units)
        for i, v in enumerate(iterable=dense_units):
            if i == 0:
                dense.append(
                    {
                        "units": v,
                        "input_shape": (x_train.shape[1],),
                        "activation": "relu",
                    }
                )
            else:
                dense.append({"units": v, "activation": "relu"})

        dense.append({"units": len(y_train[0]), "activation": "softmax"})

        model = my_tf(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dense=dense,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            reduce_lr=reduce_lr,
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            tensorboard_path=tensorboard_path,
            verbose=verbose,
            history_table=history_table,
            figsize=figsize,
            dpi=dpi,
        )

    return model
