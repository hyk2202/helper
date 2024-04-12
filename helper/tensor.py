import numpy as np

from tensorflow.random import set_seed
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import (
    History,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from pandas import DataFrame
from matplotlib import pyplot as plt
from .util import my_pretty_table
from .core import get_random_state

set_seed(get_random_state())
__initializer__ = GlorotUniform(seed=get_random_state())


def tf_create(
    dense: list = [],
    optimizer: str = "adam",
    loss: str = None,
    metrics: list = None,
    load_model: str = None,
) -> Sequential:

    if load_model:
        return load_model(load_model)

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


def tf_result(
    result: History,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
) -> Sequential:
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


def my_tf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense: list = [],
    optimizer: str = "adam",
    loss: str = None,
    metrics: list = None,
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    checkpoint_path: str = None,
    load_model: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
) -> Sequential:

    model = tf_create(dense=dense, optimizer=optimizer, loss=loss, metrics=metrics)

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


def my_tf_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    dense_units: list = [64, 32],
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
    epochs: int = 500,
    batch_size: int = 32,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    load_model: str = None,
    checkpoint_path: str = None,
    tensorboard_path: str = None,
    verbose: int = 0,
    history_table: bool = False,
    figsize: tuple = (7, 5),
    dpi: int = 100,
) -> Sequential:

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

    return my_tf(
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
        load_model=load_model,
        tensorboard_path=tensorboard_path,
        verbose=verbose,
        history_table=history_table,
        figsize=figsize,
        dpi=dpi,
    )
