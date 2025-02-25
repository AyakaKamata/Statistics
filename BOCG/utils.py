# %%
import numpy as np
from numpy.linalg import inv
import scipy.stats as ss
from itertools import islice
from __future__ import division
from typing import Tuple
from abc import ABC, abstractmethod

# ---generate data---
##normal
##multinormal
##copula
# ---hazard function---
# ---online likelihoods---
##base
##MultivariateT
##StudentT
# ---online model---


# %%
# ---generate data---
##normal
##multinormal
##copula
def generate_normal_time_series(
    num: int, minl: int = 50, maxl: int = 1000, seed: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    複数の正規分布から生成したデータを結合して1つの時系列データを作成します。

    Args:
        num (int): 生成するセグメント（部分時系列）の数。
        minl (int): 各セグメントの最小長。
        maxl (int): 各セグメントの最大長。
        seed (int): 乱数のシード（再現性確保のため）。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - partition: 各セグメントのサンプル数が格納されたNumPy配列。
            - data: 結合された時系列データを2次元（列ベクトル）に整形したNumPy配列。
    """
    # 乱数のシードを設定
    np.random.seed(seed)
    # 結合するデータを格納するための空の配列（浮動小数点型）を初期化
    data: np.ndarray = np.array([], dtype=np.float64)
    # セグメントごとのサンプル数をランダムに生成（minl以上maxl未満の整数）
    partition: np.ndarray = np.random.randint(minl, maxl, num)

    # 各セグメントに対してデータを生成
    for p in partition:
        # セグメントごとの平均値（正規分布に従い、10倍してスケール）
        mean: float = np.random.randn() * 10
        # セグメントごとの標準偏差（乱数を生成し、符号が負なら正に変換）
        var: float = np.random.randn() * 1
        if var < 0:
            var = -var
        # 指定されたサンプル数pに対して、平均mean、標準偏差varの正規分布からデータを生成
        tdata: np.ndarray = np.random.normal(mean, var, p)
        # 生成したデータをこれまでのデータに連結
        data = np.concatenate((data, tdata))

    # 1次元配列のデータを2次元の列ベクトルに変換して返す
    return partition, np.atleast_2d(data).T


def generate_multinormal_time_series(
    num: int, dim: int, minl: int = 50, maxl: int = 1000, seed: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    複数の多変量正規分布からデータを生成し、1つのデータセットとして結合します。

    Args:
        num (int): 生成するセグメント（部分時系列）の数。
        dim (int): 各多変量正規分布の次元数。
        minl (int): 各セグメントの最小サンプル数。
        maxl (int): 各セグメントの最大サンプル数。
        seed (int): 乱数のシード（再現性確保のため）。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - partition: 各セグメントのサンプル数が格納されたNumPy配列。
            - data: 各行が多変量正規分布のサンプルとなるデータ配列。
    """
    # 乱数のシードを設定
    np.random.seed(seed)
    # 後で連結するため、初期のダミー行を持つ空の2次元配列を用意（後で削除）
    data: np.ndarray = np.empty((1, dim), dtype=np.float64)
    # セグメントごとのサンプル数をランダムに生成
    partition: np.ndarray = np.random.randint(minl, maxl, num)

    for p in partition:
        # 各セグメントごとにランダムな平均ベクトルを生成（10倍してスケール）
        mean: np.ndarray = np.random.standard_normal(dim) * 10
        # ランダムな行列を生成し、その行列との積で対称正定値行列（共分散行列）を構成
        A: np.ndarray = np.random.standard_normal((dim, dim))
        cov: np.ndarray = np.dot(A, A.T)

        # 指定されたサンプル数pに対して、多変量正規分布からサンプルを生成
        tdata: np.ndarray = np.random.multivariate_normal(mean, cov, p)
        # 生成したサンプルをデータ配列に連結
        data = np.concatenate((data, tdata))

    # 初期に用意したダミー行を除去して返す
    return partition, data[1:, :]


def generate_copula(
    minl: int = 50, maxl: int = 1000, seed: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3種類の異なる相関構造を持つ2次元（bivariate）の正規分布からデータを生成し、
    それらを結合してモチベーション例を作成します。

    - 最初の分布: 正の相関を持つ (0.75)
    - 2番目の分布: 相関なし
    - 3番目の分布: 負の相関を持つ (-0.75)

    Args:
        minl (int): 各分布から生成するサンプル数の最小値。
        maxl (int): 各分布から生成するサンプル数の最大値。
        seed (int): 乱数のシード（再現性確保のため）。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - partition: 各分布から生成したサンプル数が格納されたNumPy配列（長さ3）。
            - data: 結合された全サンプルのデータ配列（各行が1サンプル）。
    """
    # 乱数のシードを設定
    np.random.seed(seed)
    dim: int = 2  # bivariate（2次元）のための次元数
    num: int = 3  # 3種類の分布を生成

    # 各分布のサンプル数をランダムに決定
    partition: np.ndarray = np.random.randint(minl, maxl, num)

    # すべての分布で共通する平均ベクトル（ゼロベクトル）
    mu: np.ndarray = np.zeros(dim)

    # 1つ目の分布: 正の相関を持つ共分散行列
    Sigma1: np.ndarray = np.array([[1.0, 0.75], [0.75, 1.0]])
    data: np.ndarray = np.random.multivariate_normal(mu, Sigma1, partition[0])

    # 2つ目の分布: 相関なしの共分散行列
    Sigma2: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
    data = np.concatenate(
        (data, np.random.multivariate_normal(mu, Sigma2, partition[1]))
    )

    # 3つ目の分布: 負の相関を持つ共分散行列
    Sigma3: np.ndarray = np.array([[1.0, -0.75], [-0.75, 1.0]])
    data = np.concatenate(
        (data, np.random.multivariate_normal(mu, Sigma3, partition[2]))
    )

    # 各分布から生成されたサンプル数と結合したデータを返す
    return partition, data


# %%
# ---hazard function---
def constant_hazard(lam, r):
    """
    Hazard function for bayesian online learning
    Arguments:
        lam - inital prob
        r - R matrix
    """
    return 1 / lam * np.ones(r.shape)


# %%
# ---online likelihoods---
##base
##MultivariateT
##StudentT
class BaseLikelihood(ABC):
    """
    This is an abstract class to serve as a template for future users to mimick
    if they want to add new models for online bayesian changepoint detection.

    Make sure to override the abstract methods to do which is desired.
    Otherwise you will get an error.

    Update theta has **kwargs to pass in the timestep iteration (t) if desired.
    To use the time step add this into your update theta function:
        timestep = kwargs['t']
    """

    @abstractmethod
    def pdf(self, data: np.array):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class to override this function."
        )

    @abstractmethod
    def update_theta(self, data: np.array, **kwargs):
        raise NotImplementedError(
            "Update theta is not defined. Please define in separate class to override this function."
        )


class MultivariateT(BaseLikelihood):
    def __init__(
        self,
        dims: int = 1,
        dof: int = 0,
        kappa: int = 1,
        mu: float = -1,
        scale: float = -1,
    ):
        """
        Create a new predictor using the multivariate student T distribution as the posterior predictive.
            This implies a multivariate Gaussian distribution on the data, a Wishart prior on the precision,
             and a Gaussian prior on the mean.
             Implementation based on Haines, T.S., Gaussian Conjugate Prior Cheat Sheet.
        :param dof: The degrees of freedom on the prior distribution of the precision (inverse covariance)
        :param kappa: The number of observations we've already seen
        :param mu: The mean of the prior distribution on the mean
        :param scale: The mean of the prior distribution on the precision
        :param dims: The number of variables
        """
        # We default to the minimum possible degrees of freedom, which is 1 greater than the dimensionality
        if dof == 0:
            dof = dims + 1
        # The default mean is all 0s
        if mu == -1:
            mu = [0] * dims
        else:
            mu = [mu] * dims

        # The default covariance is the identity matrix. The scale is the inverse of that, which is also the identity
        if scale == -1:
            scale = np.identity(dims)
        else:
            scale = np.identity(scale)

        # Track time
        self.t = 0

        # The dimensionality of the dataset (number of variables)
        self.dims = dims

        # Each parameter is a vector of size 1 x t, where t is time. Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def pdf(self, data: np.array):
        """
        Returns the probability of the observed data under the current and historical parameters
        Parmeters:
            data - the datapoints to be evaualted (shape: 1 x D vector)
        """
        self.t += 1
        t_dof = self.dof - self.dims + 1
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))
        ret = np.empty(self.t)
        try:
            # This can't be vectorised due to https://github.com/scipy/scipy/issues/13450
            for i, (df, loc, shape) in islice(
                enumerate(zip(t_dof, self.mu, inv(expanded * self.scale))), self.t
            ):
                ret[i] = ss.multivariate_t.pdf(x=data, df=df, loc=loc, shape=shape)
        except AttributeError:
            raise Exception(
                "You need scipy 1.6.0 or greater to use the multivariate t distribution"
            )
        return ret

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        centered = data - self.mu

        # We simultaneously update each parameter in the vector, because following figure 1c of the BOCD paper, each
        # parameter for a given t, r is derived from the same parameter for t-1, r-1
        # Then, we add the prior back in as the first element
        self.scale = np.concatenate(
            [
                self.scale[:1],
                inv(
                    inv(self.scale)
                    + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2))
                    * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1))
                ),
            ]
        )
        self.mu = np.concatenate(
            [
                self.mu[:1],
                (np.expand_dims(self.kappa, 1) * self.mu + data)
                / np.expand_dims(self.kappa + 1, 1),
            ]
        )
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])


class StudentT(BaseLikelihood):
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution
        https://en.wikipedia.org/wiki/Normal-gamma_distribution

        Parameters:
            alpha - alpha in gamma distribution prior
            beta - beta inn gamma distribution prior
            mu - mean from normal distribution
            kappa - variance from normal distribution
        """

        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        """
        Return the pdf function of the t distribution

        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        return ss.t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


# %%
# ---online model---
def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    https://scientya.com/bayesian-online-change-point-detection-an-intuitive-understanding-b2d2b9dc165b

    Parameters:
    data    -- the time series data

    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = R[:, t].argmax()

    return R, maxes
