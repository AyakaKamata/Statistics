# %%
import numpy as np
from numpy.linalg import inv
import scipy.stats as ss
from itertools import islice
from typing import Tuple, Optional, List
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
    np.random.seed(seed)
    data: np.ndarray = np.array([], dtype=np.float64)
    # セグメントごとのサンプル数をランダムに生成（minl以上maxl未満の整数）
    partition: np.ndarray = np.random.randint(minl, maxl, num)

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
    dim: int = 2,
    sigmas: Optional[List[np.ndarray]] = None,
    minl: int = 50,
    maxl: int = 1000,
    seed: int = 100,
    mu: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    任意の次元（dim）と各分布の共分散行列（sigmas）を指定して、
    複数の多変量正規分布からサンプルを生成し、結合します。
    Args:
        dim (int): 次元数。muがNoneの場合は、ゼロベクトルが平均として用いられます。
        sigmas (Optional[List[np.ndarray]]): 各分布の共分散行列のリスト。
        minl (int): 各分布から生成するサンプル数の最小値。
        maxl (int): 各分布から生成するサンプル数の最大値。
        seed (int): 乱数シード（再現性のため）。
        mu (Optional[np.ndarray]): 平均ベクトル。Noneの場合、ゼロベクトル (長さdim) が用いられる。
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - partition: 各分布から生成したサンプル数を格納した配列（長さは分布数）。
            - data: 結合された全サンプルのデータ（各行が1サンプル）。
    """
    np.random.seed(seed)

    # 平均ベクトルの設定（指定がなければゼロベクトル）
    if mu is None:
        mu = np.zeros(dim)

    # 共分散行列の既定値設定（dim==2の場合のみ）
    if sigmas is None:
        raise ValueError("sigmas is None")

    num = len(sigmas)  # 分布の数
    partition: np.ndarray = np.random.randint(minl, maxl, num)

    data_list = []
    for i in range(num):
        samples = np.random.multivariate_normal(mu, sigmas[i], partition[i])
        data_list.append(samples)

    data: np.ndarray = np.concatenate(data_list, axis=0)
    return partition, data


# %%
# ---hazard function---
def constant_hazard(lam, r):
    """
    ベイズオンライン学習における一定ハザード関数。
    引数:
        lam: 成功するまでに必要な試行回数の期待値
        r: R行列
    戻り値:
        rと同じ形状で、全ての要素が1/lamの配列
    """
    return 1 / lam * np.ones(r.shape)


# %%
# ---online likelihoods---
##base
##MultivariateT(Marginal Distribution of x)
##StudentT(Marginal Distribution of x)


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
        x|mu,Sigma^-1~N_dims(mu,Sigma^-1)
        mu|Sigma^-1~N(mu_0,(kappa_0*Sigma^-1)^-1)
        Sigma^-1~Wishart
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
        観測データ data に対する、現在および過去のパラメータ下での確率密度関数 (PDF) の値を返す関数
        引数:
            data: 評価対象のデータポイント (shape: 1 x D のベクトル)
        """
        # 時刻 t を1増やす
        self.t += 1

        # 有効な自由度の計算
        t_dof = self.dof - self.dims + 1

        # スケールパラメータの補正係数を計算
        # (self.kappa * t_dof) / (self.kappa + 1) を計算し、軸1と2(0-index)を新たに追加して次元を合わせる
        expanded = np.expand_dims((self.kappa * t_dof) / (self.kappa + 1), (1, 2))

        ret = np.empty(self.t)

        try:
            # scipy の multivariate_t.pdf はベクトル化できないため、各時刻のパラメータに対して個別に計算する
            # islice を使って self.t 個分のパラメータ (自由度, 平均, スケール行列の逆行列) を反復処理
            for i, (df, loc, shape) in islice(
                enumerate(zip(t_dof, self.mu, inv(expanded * self.scale))), self.t
            ):
                # multivariate_t.pdf を用いて、data の確率密度を計算し ret 配列に格納
                ret[i] = ss.multivariate_t.pdf(x=data, df=df, loc=loc, shape=shape)
        except AttributeError:
            raise Exception(
                "You need scipy 1.6.0 or greater to use the multivariate t distribution"
            )
        return ret


def update_theta(self, data: np.array, **kwargs):
    """
    観測データ data に基づいて、事前パラメータ (theta) のベイズ更新を行う関数
    引数:
        data: 評価対象のデータポイント (shape: 1 x D のベクトル)
    """
    # 観測データと現在の平均 (self.mu) との差（中心化された値）を計算
    centered = data - self.mu

    # 各更新時刻において、前時刻のパラメータから派生して更新を行い、
    # 最初の（事前）パラメータはそのまま保持するために先頭の要素を残して連結する

    # self.scale (共分散行列) の更新:
    # 1. 先頭の要素（事前のスケール）はそのまま保持する
    # 2. 以降の要素は、元のスケール行列の逆行列に、データの中心化項の外積に係数 (kappa/(kappa+1)) を掛けたものを加えた後、
    #    その逆行列を取ることで更新する
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

    # self.mu (平均ベクトル) の更新:
    # 1. 先頭の要素（事前の平均）はそのまま保持する
    # 2. 以降の要素は、重み付き平均として、古い平均に kappa を掛けたものと新たなデータを足し合わせ、
    #    (kappa+1) で割ることで更新する
    self.mu = np.concatenate(
        [
            self.mu[:1],
            (np.expand_dims(self.kappa, 1) * self.mu + data)
            / np.expand_dims(self.kappa + 1, 1),
        ]
    )

    # 自由度 (self.dof) の更新:
    # 1. 先頭の要素はそのまま保持する
    # 2. 以降は全て 1 を加算する
    self.dof = np.concatenate([self.dof[:1], self.dof + 1])

    # kappa の更新:
    # 1. 先頭の要素はそのまま保持する
    # 2. 以降は全て 1 を加算する
    self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])


class StudentT(BaseLikelihood):
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):
        """
        x|sigma^2~N(mu,sigma^2)
        mu|sigma^2~N(mu_0,sigma^2/kappa_0)
        sigma^-2~Gamma
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
# Adams and MacKay 2007
def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    Parameters:
    data    -- the time series data
    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    # enumerateでt=0~(T-1)なので、表記上は１を足している
    for t, x in enumerate(data):
        # 3.Evaluage Predictive Probability
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.
        predprobs = log_likelihood_class.pdf(x)

        # (prepare for 4 and 5)Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # 4.Calculate Growth Probabilities
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        # R[0 : t + 1, t]=Message
        # dim: ((t+1)*1)*(1)*(1*(t+1))=(t+1)*(t+1)
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # 5.Calculate Changepoint Probabilities
        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        # dim: ((t+1)*1)*(1)*(1*(t+1))=(t+1)*(t+1)の全要素の合計
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # 6,7.Calculate Evidence & Determine Run Length Distribution
        # r_{t}の全部の場合のrun length probを求める
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])
        if R.shape[1] >= 1 and R[1, t + 1] >= 0.95:
            print(f"changepoint detected at time step {t+1}")
        # 8,9.Update Sufficient Statistics & Perform Prediction
        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = R[:, t].argmax()

    return R, maxes
