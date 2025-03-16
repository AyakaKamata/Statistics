import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import util


class Perceptron_Classifier:
    """
    Perceptron Binary Classifier
    ---
    params
    ---
    eta: float
      learning rate (from 0.0 to 1.0)
    n_iter: int
      training iteration
    seed: int
      random seed

    attributes
    ---
    w\_: onevecs
      posterior weight
    b\_: scalar
      posterior intercept
    errors\_: list
      collect error numbers at each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):
        """
        params
        ---
        X: NumPy.ndarray, shape=[n_examples, n_features]
          train data
        y: Numpy.ndarray, shape=[n_examples]
          objective

        returns
        ---
        self: object
        """
        # generate random variable
        # set seed
        rgen = np.random.RandomState(self.seed)
        # generate small numbers, size=[n_features]
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # initiate intercept as 0
        self.b_ = np.float64(0.0)
        # errors_ is empty
        self.erros_ = []
        for _ in range(self.n_iter):
            errors = 0
            # update weight and intercept
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                # if predicted value is not true, add 1
                errors += int(update != 0.0)
            self.erros_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate input
        ---
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        Return prediction
        ---
        """
        # if input is larger that 0, return 1, otherwise return 0
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def plot_errors(self):
        """Plot loss values over epochs."""
        sns.set_theme()
        sns.relplot(
            x=range(1, len(self.erros_) + 1), y=self.erros_, kind="line", marker="o"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Training Error over Epochs")
        plt.show()


class ADALINE_Classifier:
    """
    Adaptive Linear Neuron
    -----------------
    use MSE method

    Parameters:
      eta: float
          Learning rate (from 0.0 to 1.0)
      n_iter: int
          Number of iterations (epochs)
      seed: int
          Random seed for reproducibility

    Attributes:
      w_: ndarray
          Weight vector
      b_: float
          Bias term
      losses_: list
          MSE at each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):
        """
        params
        ---
        X: Polars.DataFrame, shape=[n_examples, n_features]
          train data
        y: Polars.DataFrame, shape=[n_examples]
          objective

        returns
        ---
        self: object
        """
        # set seed
        rgen = np.random.RandomState(self.seed)
        # generate small numbers, size=[n_features]
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # initiate intercept as 0
        self.b_ = np.float64(0.0)
        # errors_ is empty
        self.losses_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activate(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """
        Calculate input
        ---
        """
        return np.dot(X, self.w_) + self.b_

    def activate(self, X):
        """
        Calculate activation function
        ---
        """
        return X

    def predict(self, X):
        # if net_input is larger that 0.5, return 1, otherwise return 0
        return np.where(self.activate(self.net_input(X)) >= 0.5, 1, 0)

    def plot_losses(self):
        """Plot loss values over epochs."""
        sns.set_theme()
        sns.relplot(
            x=range(1, len(self.losses_) + 1), y=self.losses_, kind="line", marker="o"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.show()


class ADALINE_Classifier_SGD:
    """
    ADALINE SGD ver.
    -----------------
    Parameters:
      eta: float
          Learning rate (from 0.0 to 1.0)
      n_iter: int
          Number of iterations (epochs)
      shuffle: bool(=True)
        Shuffle training data at each start of epochs
      seed: int
          Random seed for reproducibility

    Attributes:
      w_: ndarray
          Weight vector
      b_: float
          Bias term
      losses_: list
          List to store the loss (misclassifications) in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, seed=1):
        self.eta = eta
        self.n_iter = n_iter
        # set w_init flag to false
        self.w_initialized = False
        self.shuffle = shuffle
        self.seed = seed

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : Polars.DataFrame, shape=[n_examples, n_features]
            Training data
        y : Polars.DataFrame, shape=[n_examples]
            Objective

        Returns
        -------
        self : object
        """

        self._initialize_w_b_(X.shape[1])
        # errors_ is empty
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, yi in zip(X, y):
                xi = np.array(xi)
                losses.append(self._update_w_b_(xi, yi))
            # calculate average loss of training data at this epoch
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def partial_fit(self, X, y):
        """
        Updatae w_ and b_ with new data
        """
        if not self.w_initialized:
            self._initialize_w_b_(X.shape[1])

        # if the size of all data y > 2, update w_ and b_ with xi and yi
        if y.ravel().shape[0] > 1:
            for xi, yi in zip(X, y):
                self._update_w_b_(xi, yi)
        else:
            self._update_w_b_(X, y)
        return self

    def _initialize_w_b_(self, w_dim):
        """Initialize w\_ and b\_"""
        # set seed
        self.rgen = np.random.RandomState(self.seed)
        # generate small numbers, size=[n_features]
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=w_dim)
        # initiate intercept as 0
        self.b_ = np.float64(0.0)
        self.w_initialized = True

    def _shuffle(self, X, y):
        """Shuffle training Data after _initialize_w_b_"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_w_b_(self, xi, yi):
        """Update w\_ and b\_"""
        output = self.activate(self.net_input(xi))
        error = yi - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    def net_input(self, X):
        """Compute the net input (weighted sum plus bias)"""
        return np.dot(X, self.w_) + self.b_

    def activate(self, X):
        """
        Calculate activation function
        """
        return X

    def predict(self, X):
        # if net_input is larger that 0.5, return 1, otherwise return 0
        return np.where(self.activate(self.net_input(X)) >= 0.5, 1, 0)

    def plot_losses(self):
        """Plot loss values over epochs."""
        sns.set_theme()
        sns.relplot(
            x=range(1, len(self.losses_) + 1), y=self.losses_, kind="line", marker="o"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.show()


def plot_decision_regions(X, y, classifier, grid=0.02):
    """
    Plot decision regions using classifier
    ---
    **classifier**: already fitted, must have predict function
    """

    # 1. 決定領域を描画するためのメッシュを作成
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))

    # 2. メッシュ上の各点を分類器で予測
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 3. 等高線で決定境界を可視化
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Pastel1")
    # 4. 実際のデータ点を Seaborn の散布図で重ねて描画
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,  # クラスを色分け
    )


def standardize(arr):
    """
    標準化を各列ごとに適用する関数。

    Parameters:
    arr (np.ndarray): 標準化したい2次元のNumPy配列

    Returns:
    np.ndarray: 各列が平均0, 標準偏差1にスケールされた配列
    """
    mean = np.mean(arr, axis=0)  # 各列の平均
    std = np.std(arr, axis=0)  # 各列の標準偏差 (母標準偏差)

    # 標準偏差が0の列はそのままにする (ゼロ除算回避)
    standardized = (arr - mean) / std

    return standardized
