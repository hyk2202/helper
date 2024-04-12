import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History, EarlyStopping, ReduceLROnPlateau
from pandas import DataFrame
from matplotlib import pyplot as plt
from .util import my_pretty_table


def tf_create(
    dense: list = [],
    optimizer: str = "adam",
    loss: str = None,
    metrics: list = None,
) -> Sequential:

    if not dense or not loss or not metrics:
        raise ValueError("dense, loss, and metrics are required arguments")

    model = Sequential()

    for i, v in enumerate(iterable=dense):
        if "input_shape" in v:
            model.add(
                Dense(
                    units=v["units"],
                    input_shape=v["input_shape"],
                    activation=v["activation"],
                )
            )
        else:
            model.add(Dense(units=v["units"], activation=v["activation"]))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def tf_train(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray = None,
    y_test: np.ndarray = None,
    epochs: int = 500,
    patience: int = 10,
    batch_size: int = 32,
    factor: float = 0.1,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    verbose: int = 0,
) -> History:

    callbacks = []

    if early_stopping:
        callbacks.append(
            EarlyStopping(patience=patience, restore_best_weights=True, verbose=verbose)
        )

    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(factor=factor, patience=patience // 2, verbose=verbose)
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


def tf_result(result: History, figsize: tuple = (7, 5), dpi: int = 100) -> dict:
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

    my_pretty_table(result_df)
