# %%
import math
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.infer.reparam import LinearHMMReparam, StableReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat
from pyro.ops.stats import quantile
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy

pyro.set_rng_seed(20200305)
# %%
# load data
dataset = torch.tensor(pd.read_csv("nile.csv", index_col=0).values).float()
plt.plot(dataset)
plt.show()
# %%
plt.acorr(dataset.flatten(), maxlags=10)
plt.show()
# %%
T0 = 0  # beginning
T2 = dataset.size(-2)  # end
T1 = T2 - 7 * 2  # train/test split
means = dataset.mean(0)
covariates = torch.zeros(len(dataset), 0)  # empty


# %%
# define model
class Model1(ForecastingModel):
    def model(self, zero_data, covariates):
        duration = zero_data.size(-2)

        # We'll hard-code the periodic part of this model, learning only the local model.
        prediction = periodic_repeat(means, duration, dim=-1).unsqueeze(-1)

        # On top of this mean prediction, we'll learn a linear dynamical system.
        # This requires specifying five pieces of data, on which we will put structured priors.
        init_dist = dist.Normal(0, 10).expand([1]).to_event(1)

        timescale = pyro.sample("timescale", dist.LogNormal(math.log(24), 1))
        # Note timescale is a scalar but we need a 1x1 transition matrix (hidden_dim=1),
        # thus we unsqueeze twice using [..., None, None].
        trans_matrix = torch.tensor([1.0])[..., None, None]
        trans_scale = pyro.sample("trans_scale", dist.LogNormal(-0.5 * math.log(24), 1))
        trans_dist = dist.Normal(0, trans_scale.unsqueeze(-1)).to_event(1)

        # Note the obs_matrix has shape hidden_dim x obs_dim = 1 x 1.
        obs_matrix = torch.tensor([[1.0]])
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(-2, 1))
        obs_dist = dist.Normal(0, obs_scale.unsqueeze(-1)).to_event(1)

        noise_dist = dist.GaussianHMM(
            init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist, duration=duration
        )
        self.predict(noise_dist, prediction)


pyro.render_model(
    Model1(),
    model_args=(dataset[:T1], covariates[:T1]),
    render_params=True,
    render_distributions=True,
    render_deterministic=True,
)
# %%
pyro.set_rng_seed(1)
pyro.clear_param_store()
start = time.time()  # 現在時刻（処理開始前）を取得

forecaster = Forecaster(
    Model1(),
    dataset[:T1],
    covariates[:T1],
    learning_rate=0.1,
    learning_rate_decay=0.05,
    num_steps=1000,
)
end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print("time(mins):", time_diff)  # 処理にかかった時間データを使用

for name, value in forecaster.guide.median().items():
    if value.numel() == 1:
        print("{} = {:0.4g}".format(name, value.item()))
# %%
# ---Plot the training losses---
plt.figure(figsize=(10, 5))
plt.plot(forecaster.losses[300:], label="Training Loss")
plt.xlabel("SVI step after 300 iter")
plt.ylabel("ELBO loss")
plt.show()
# %%
samples = forecaster(dataset[:T1], covariates, num_samples=100)
p10, p50, p90 = quantile(samples, (0.1, 0.5, 0.9)).squeeze(-1)
crps = eval_crps(samples, dataset[T1:])
print(samples.shape, p10.shape)

plt.figure(figsize=(9, 3))
plt.fill_between(torch.arange(T1, T2), p10, p90, color="red", alpha=0.3)
plt.plot(torch.arange(T1, T2), p50, "r-", label="forecast")
plt.plot(torch.arange(T1 - 7 * 2, T2), dataset[T1 - 7 * 2 : T2], "k-", label="truth")
plt.title("CRPS = {:0.3g}".format(crps))
plt.xlim(T1 - 7 * 2, T2)
plt.legend(loc="best")
