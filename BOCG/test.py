# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import partial
from utils import (
    generate_normal_time_series,
    generate_copula,
    constant_hazard,
    online_changepoint_detection,
    StudentT,
    MultivariateT,
)

# %%
# ---visualize generated data---
partition, data = generate_normal_time_series(7, 50, 200)
fig, ax = plt.subplots(figsize=[16, 12])
ax.plot(data)
# %%
# ---prior of r---
# constant_hazard の第1引数を 250 に固定して新しい関数を作成する。
# これにより、hazard_function(R) を呼び出すと、内部的には constant_hazard(250, R) が実行される。
hazard_function = partial(constant_hazard, 250)

# %%
# ---calculate growth probability value, max---
R, maxes = online_changepoint_detection(
    data,
    hazard_function,
    StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0),
)
# %%
# ---visualize data,growth probability value,max---
epsilon = 1e-7
fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
ax[0].plot(data)
sparsity = 5  # only plot every fifth data for faster display
density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity] + epsilon)
ax[1].pcolor(
    np.array(range(0, len(R[:, 0]), sparsity)),
    np.array(range(0, len(R[:, 0]), sparsity)),
    density_matrix,
    cmap=cm.Greys,
    vmin=0,
    vmax=density_matrix.max(),
    shading="auto",
)
# 各t時点での、run lengthが1以上となる確率
Nw = 1
ax[2].plot(R[Nw, Nw:-1])

# %%
# ---Copula---
# ---Sigma---
dim = 4
r = 0.8

# k=1 の SIGMA: 対角成分が 1,1,1,1 の対角行列
sigma1 = np.diag([1, 1, 1, 1])

# k=2 の SIGMA: 全要素が 0.5 の行列に対角成分0.5を加えるので、対角は 1, 非対角は 0.5 となる
sigma2 = np.full((dim, dim), 0.5) + np.diag(np.full(dim, 0.5))

# k=3 の SIGMA: 行ごとに r のべき乗で作成（例: 1行目は [r^0, r^1, r^2, r^3]）
sigma3 = np.array(
    [
        [r**0, r**1, r**2, r**3],
        [r**1, r**0, r**1, r**2],
        [r**2, r**1, r**0, r**1],
        [r**3, r**2, r**1, r**0],
    ]
)

# Python の List[ndarray] としてまとめる
SIGMA = [sigma1, sigma2, sigma3, sigma2]

# ---visualize generated data---
mu = np.array([12, 9, 7, 5])
partition, data = generate_copula(dim, SIGMA, 50, 100, 100, mu)
fig, ax = plt.subplots(figsize=[16, 12])
tmp = 0
for p in partition:
    tmp += p
    ax.axvline(x=tmp, color="b", linestyle="--")
    print(tmp)
ax.plot(data)
plt.show()
print(SIGMA)
# %%
# ---prior of r---
# constant_hazard の第1引数を 250 に固定して新しい関数を作成する。
# これにより、hazard_function(R) を呼び出すと、内部的には constant_hazard(250, R) が実行される。
hazard_function = partial(constant_hazard, 100)

# %%
# ---calculate growth probability value, max---
# t=300で5秒ぐらい
R, maxes, dict = online_changepoint_detection(
    data,
    hazard_function,
    MultivariateT(dims=dim, mu=8),
)
# %%
# ---visualize data,growth probability value,max---
epsilon = 1e-7
fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
tmp = 0
for p in partition:
    tmp += p
    ax[0].axvline(x=tmp, color="b", linestyle="--")
ax[0].plot(data)
sparsity = 5  # only plot every fifth data for faster display
density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity] + epsilon)
ax[1].pcolor(
    np.array(range(0, len(R[:, 0]), sparsity)),
    np.array(range(0, len(R[:, 0]), sparsity)),
    density_matrix,
    cmap=cm.Greys,
    vmin=0,
    vmax=density_matrix.max(),
    shading="auto",
)
# 各t時点での、run lengthが1以上となる確率
Nw = 1
ax[2].plot(R[Nw, Nw:-1])
print(dict)

# %%
