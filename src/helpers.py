import numpy as np
import matplotlib.pyplot as plt

from random import shuffle


def quick_plot_img(data, x_dim=80, y_dim=80, layer_dim=3, order="F"):
    """
    Helper, plot image quickly, assumes 1D input or array or list.
    The default values are all for this specific project's dataset.
    """
    img = np.reshape(np.array(data), (x_dim, y_dim, layer_dim), order=order)
    plt.imshow((img))


def format_imgs(data, x_dim=80, y_dim=80, layer_dim=3, order="F"):
    """Helper, reshape 1D input data into an appropriate numpy array."""
    return np.reshape(
        np.array(data), (len(data), x_dim, y_dim, 3), order=order
    )  # noqa:E501


def quick_plot_imggen(img, datagen, n_examples=6):
    img = img.reshape((1,) + img.shape)
    i = 0

    # Organise some subplots
    fig, axs = plt.subplots(2, int(np.ceil(n_examples / 2)))

    for batch in datagen.flow(img, batch_size=1):
        # Dump augmented image to a subplot
        x = i % 2
        y = i % 3
        axs[x][y].imshow((batch[0]))

        # Iterate
        i += 1
        if i % n_examples == 0:
            break

    plt.show()


def train_test_validation_split(
    X: iter,
    y: iter,
    train_size: float = 0.8,
    validation=True,
    data_cap: int = None,  # noqa:E501
):
    """
    Shuffles and splits features and labels at random into training, validation
    and test sets.

    Args:
        X (iter): Features iterable
        y (iter): Labels iterable
        train_size (float, optional): Fraction to end up in training set.
            Defaults to 0.8.
        validation (bool, optional): Wether to split the test set further
            into test and validation. Defaults to True.
        data_cap (int, optional): limit on how much data to create total

    Returns:
        tuple: training, validation and test features and labels.
    """

    # Shuffle the data at random
    data = list(zip(X, y))
    shuffle(data)
    if data_cap:
        data = data[:data_cap]
    X, y = zip(*data)

    train_X = None
    train_y = None
    val_X = None
    val_y = None
    test_X = None
    test_y = None

    train_cutoff = int(len(y) * train_size)
    train_X = X[:train_cutoff]
    train_y = y[:train_cutoff]

    # If there's a validation set then need to further split the non-train data
    if validation:
        val_cutoff = int(len(y) * ((train_size + 1.0) / 2))
        val_X = X[train_cutoff:val_cutoff]
        val_y = y[train_cutoff:val_cutoff]
        test_X = X[val_cutoff:]
        test_y = y[val_cutoff:]

    # Otherwise just create a test set
    else:
        test_X = X[train_cutoff:]
        test_y = y[train_cutoff:]

    # Report the result - it's random so someone might want to retry
    stats = (
        f"train: {len(train_X)}, "
        + f"validation: {len(val_X)}, "
        + f"test: {len(test_X)}"
    )
    print(stats)

    if val_X:
        return (train_X, train_y, val_X, val_y, test_X, test_y)
    return (train_X, train_y, test_X, test_y)
