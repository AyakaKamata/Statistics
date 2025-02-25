# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import partial
from __future__ import division
from utils import (
    generate_normal_time_series,
    constant_hazard,
    online_changepoint_detection,
    StudentT,
)

# %%
# ---visualize generated data---
partition, data = generate_normal_time_series(7, 50, 200)
fig, ax = plt.subplots(figsize=[16, 12])
ax.plot(data)
# %%
# ---prior of r---
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
Nw = 10
ax[2].plot(R[Nw, Nw:-1])
