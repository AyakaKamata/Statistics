# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.infer.reparam import LocScaleReparam
from pyro.ops.stats import quantile

# %%
# ---setting---
pyro.set_rng_seed(20200928)

pd.set_option("display.max_rows", 500)
plt.style.use("fivethirtyeight")
# https://pyro.ai/examples/forecasting_dlm.html

# %%
# ---generate observation data---
"""
univariate DLM
y_t = beta_t[0]+t(x_t(covariates)[1:p+1]) * beta_t[1:p+1] + epsilon_t, epsilon_t-N(0, sigma_e^2)
beta_t = beta_{t-1} + delta_t, delta_t-N(0, sigma_d^2)
"""
torch.manual_seed(20200101)
# number of predictors, total observations
p = 5  # x_t の次元(実際はp+1)
n = 365 * 3  # 3years

# start, train end, test end
T0 = 0
T1 = n - 28
T2 = n

# initializing coefficients(係数) at zeros, simulate all coefficient values
# shape (n, 1), sampled from N(0,0.1), cumulatively summed
beta0 = torch.empty(n, 1).normal_(0, 0.1).cumsum(0)
# shape (n, p), sampled from N(0, 0.02), cumulatively summed
betas_p = torch.empty(n, p).normal_(0, 0.02).cumsum(0)
# concatenation of beta0 and betas_p, shape (n, 1 + p).
betas = torch.cat([beta0, betas_p], dim=-1)
print("betas", betas)

# simulate regressors
# 共変量(時間依存) n*(p+1), drawn from N(0, 0.1)
covariates = torch.cat([torch.ones(n, 1), torch.randn(n, p) * 0.1], dim=-1)

# observation with noise shape(n*1), N(0, 0.1)
y = ((covariates * betas).sum(-1) + 0.1 * torch.randn(n)).unsqueeze(-1)
# %%
# ---visualize observation and betas---
fig, axes = plt.subplots(p + 2, 1, figsize=(10, 3 * (p + 2)))
for idx, ax in enumerate(axes):
    if idx == 0:
        axes[0].plot(y, "k-", label="truth", alpha=0.8)
        axes[0].legend()
        axes[0].set_title("response")
    else:
        axes[idx].plot(betas[:, idx - 1], "k-", label="truth", alpha=0.8)
        axes[idx].set_title("true_coef_{}".format(idx - 1))
plt.tight_layout()

# %%
# ---visualize covariates---
fig, axes = plt.subplots(p + 1, 1, figsize=(10, 3 * (p + 2)))
for idx, ax in enumerate(axes):
    axes[idx].plot(covariates[:, idx], "k-", label="truth", alpha=0.8)
    axes[idx].set_title("covariates_{}".format(idx))
plt.tight_layout()


# %%
# Case 1:
# ---define model---
class DLM(ForecastingModel):
    def model(self, zero_data, covariates):
        # データの次元数を取得
        # data_dim = zero_data.size(-1)
        feature_dim = covariates.size(-1)  # p+1

        # delta_tのreparam用のスケールを対数正規分布からサンプリング
        drift_scale = pyro.sample(
            "drift_scale",
            dist.LogNormal(loc=-10, scale=10).expand([feature_dim]).to_event(1),
        )

        # 時間軸に沿ってプレートを作成（時系列モデルのため）
        with self.time_plate:
            # LocScaleReparam を使用してドリフトパラメータを再パラメータ化
            # N(0,e^z),z-N(-10,10^2)
            with poutine.reparam(config={"drift": LocScaleReparam()}):
                drift = pyro.sample(
                    "drift",
                    # covariatesと同じサイズ
                    dist.Normal(
                        loc=torch.zeros(covariates.size()), scale=drift_scale
                    ).to_event(1),
                )

        # ドリフトの累積和を取ることで、beta_tのブラウン運動（ランダムウォーク）を生成
        # 列ごとに累積和（ここでは１列のみ）
        weight = drift.cumsum(-2)  # A Brownian motion.

        # 生成された重みを記録（トレース可能な値として保存）
        pyro.deterministic("weight", weight)

        # 予測値の計算（重みと共変量の積を合計）
        prediction = (weight * covariates).sum(-1, keepdim=True)
        assert prediction.shape[-2:] == zero_data.shape

        # 予測値を記録（トレース可能な値として保存）
        pyro.deterministic("prediction", prediction)

        # ノイズのスケールを対数正規分布からサンプリング
        scale = pyro.sample(
            "noise_scale", dist.LogNormal(-5, 10).expand([1]).to_event(1)
        )

        # ノイズ分布を正規分布として定義
        noise_dist = dist.Normal(0, scale)

        # 予測値にノイズを加えて最終的な予測を行う
        self.predict(noise_dist, prediction)


pyro.render_model(
    DLM(),
    model_args=(
        y[:T1],
        covariates[:T1],
    ),
    render_params=True,
    render_deterministic=True,
)

# %%
# ---train and validate---
# seed が大事らしい
pyro.set_rng_seed(1)
pyro.clear_param_store()
model = DLM()
start = time.time()  # 現在時刻（処理開始前）を取得
# variational inference
"""loss = -elbo / data.numel() ELBO（モデルの対数周辺尤度の下限）"""
forecaster = Forecaster(
    model=model,
    # train data
    data=y[:T1],
    covariates=covariates[:T1],
    # :class:`~pyro.optim.optim.DCTAdam`(Discrete Cosine Transform-augmented)
    learning_rate=0.1,
    learning_rate_decay=0.05,
    num_steps=1000,
)
end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print("time(mins):", time_diff)  # 処理にかかった時間データを使用

# %%
# ---sample from posterior---
# record all latent variables in a trace
with poutine.trace() as tr:
    forecaster(y[:T1], covariates[:T1], num_samples=100)

# extract the values from the recorded trace
posterior_samples = {
    name: site["value"]
    for name, site in tr.trace.nodes.items()
    if site["type"] == "sample"
}

# %%
# ---visualize---
"""
not all coefficients can be recovered in the vanilla model
"""
# overlay estimations with truth
fig, axes = plt.subplots(p + 2, 1, figsize=(10, 3 * (p + 2)))
# posterior quantiles of latent variables
pred_p10, pred_p50, pred_p90 = quantile(
    posterior_samples["prediction"], (0.1, 0.5, 0.9)
).squeeze(-1)
# posterior quantiles of latent variables
coef_p10, coef_p50, coef_p90 = quantile(
    posterior_samples["weight"], (0.1, 0.5, 0.9)
).squeeze(-1)

for idx, ax in enumerate(axes):
    if idx == 0:
        axes[0].plot(y[:T1], "k-", label="truth", alpha=0.8, lw=1)
        axes[0].plot(pred_p50, "r-", label="estimate", alpha=0.8, lw=1)
        axes[0].fill_between(
            torch.arange(0, T1), pred_p10, pred_p90, color="red", alpha=0.3
        )
        axes[0].legend()
        axes[0].set_title("response")
    else:
        axes[idx].plot(betas[:T1, idx - 1], "k-", label="truth", alpha=0.8, lw=1)
        axes[idx].plot(coef_p50[:, idx - 1], "r-", label="estimate", alpha=0.8, lw=1)
        axes[idx].fill_between(
            torch.arange(0, T1),
            coef_p10[:, idx - 1],
            coef_p90[:, idx - 1],
            color="red",
            alpha=0.3,
        )
        axes[idx].set_title("coef_{}".format(idx - 1))
plt.tight_layout()
# %%
# ---predict and visualize---
pyro.set_rng_seed(1)
samples = forecaster(y[:T1], covariates, num_samples=1000)
p10, p50, p90 = quantile(samples, (0.1, 0.5, 0.9)).squeeze(-1)
crps = eval_crps(samples, y[T1:])

plt.figure(figsize=(20, 5))
plt.fill_between(torch.arange(T1, T2), p10, p90, color="red", alpha=0.3)
plt.plot(torch.arange(T1, T2), p50, "r-", label="forecast", alpha=0.8)
plt.plot(np.arange(T1 - 200, T2), y[T1 - 200 : T2], "k-", label="truth", alpha=0.5)
plt.title("Response against time (CRPS = {:0.3g})".format(crps))
plt.axvline(T1, color="b", linestyle="--")
plt.legend(loc="best")
# %%
# Case 2:
# ---Use Prior---
# let's provide some priors
"""
y_t = beta_t[0]+t(x_t(covariates)[1:p+1]) * beta_t[1:p+1] + epsilon_t, epsilon_t-N(0, sigma_e^2)
beta_t = beta_{t-1} + delta_t, delta_t-N(B_t, sigma_d^2)
B_t-N(0, 0.3)
"""
time_points = np.concatenate(
    (
        np.arange(300, 320),
        np.arange(600, 620),
        np.arange(900, 920),
    )
)
# broadcast on time-points
priors = betas[time_points, 1:]

print(time_points.shape, priors.shape)


# %%
# ---define Model using Prior
class DLM2(ForecastingModel):
    def model(self, zero_data, covariates):
        data_dim = zero_data.size(-1)
        feature_dim = covariates.size(-1)

        delta_scale = pyro.sample(
            "delta_scale", dist.LogNormal(-10, 10).expand([feature_dim]).to_event(1)
        )
        with self.time_plate:
            with poutine.reparam(config={"delta": LocScaleReparam()}):
                delta = pyro.sample(
                    "delta",
                    dist.Normal(torch.zeros(covariates.size()), delta_scale).to_event(
                        1
                    ),
                )
        beta_t = delta.cumsum(-2)  # A Brownian motion.
        # record in model_trace
        pyro.deterministic("beta_t", beta_t)

        # This is the only change from the simpler DLM model.
        # We inject prior terms as if they were likelihoods using pyro observe statements.
        for tp, prior in zip(time_points, priors):
            # 指定された時間点 `tp` ごとに、対応する事前分布 `prior` を設定してサンプリング
            pyro.sample(
                "delta_prior_mean_{}".format(
                    tp
                ),  # サンプル名（各時間点ごとに異なる名前）
                # 分散が0.4以上だと大きすぎる
                dist.Normal(prior, 0.3).to_event(
                    1
                ),  # 事前分布として Normal(prior, 0.5) を使用
                obs=beta_t[
                    ..., tp : tp + 1, 1:
                ],  # 実際の `weight` の対応する部分を観測データとして与える
                # `weight[..., tp : tp + 1, 1:]` の説明:
                # - `...` → 前のすべての次元を保持（batch_size など）
                # - `tp : tp + 1` → `tp` の時点のデータを取得するが、次元を落とさない（形を維持）
                # - `1:` → 最後の次元のインデックス `1` 以降を取得
                # → `weight` の形を崩さずに Pyro に観測データとして適用するための処理
            )

        prediction = (beta_t * covariates).sum(-1, keepdim=True)
        assert prediction.shape[-2:] == zero_data.shape
        # record in model_trace
        pyro.deterministic("prediction", prediction)

        scale = pyro.sample(
            "noise_scale", dist.LogNormal(-5, 10).expand([1]).to_event(1)
        )
        noise_dist = dist.Normal(0, scale)
        self.predict(noise_dist, prediction)


pyro.render_model(
    DLM2(),
    model_args=(y[:T1], covariates[:T1]),
    render_params=True,
    render_deterministic=True,
)

# %%
# ---train and validate---
pyro.set_rng_seed(1)
pyro.clear_param_store()
model2 = DLM2()
start = time.time()  # 現在時刻（処理開始前）を取得
# variational inference
forecaster2 = Forecaster(
    model=model2,
    # train data
    data=y[:T1],
    covariates=covariates[:T1],
    # :class:`~pyro.optim.optim.DCTAdam`(Discrete Cosine Transform-augmented)
    learning_rate=0.1,
    learning_rate_decay=0.05,
    num_steps=1000,
)
end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print("time(mins):", time_diff)  # 処理にかかった時間データを使用
# %%
# ---Plot the training losses---
plt.figure(figsize=(10, 5))
plt.plot(forecaster2.losses[300:], label="Training Loss")
plt.xlabel("SVI step after 300 iter")
plt.ylabel("ELBO loss")
plt.show()
# ---print best param---
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())
# %%
# ---sample from posterior---
# record all latent variables in a trace
pyro.set_rng_seed(1)
with poutine.trace() as tr:
    forecaster2(y[:T1], covariates[:T1], num_samples=100)

# extract the values from the recorded trace
posterior_samples2 = {
    name: site["value"]
    for name, site in tr.trace.nodes.items()
    if site["type"] == "sample"
}
# %%
# ---visualize---
# overlay estimations with truth
fig, axes = plt.subplots(p + 2, 1, figsize=(10, 3 * (p + 2)))
# posterior quantiles of latent variables
pred_p10, pred_p50, pred_p90 = quantile(
    posterior_samples2["prediction"], (0.1, 0.5, 0.9)
).squeeze(-1)
# posterior quantiles of latent variables
coef_p10, coef_p50, coef_p90 = quantile(
    posterior_samples2["beta_t"], (0.1, 0.5, 0.9)
).squeeze(-1)

for idx, ax in enumerate(axes):
    if idx == 0:
        axes[0].plot(y[:T1], "k-", label="truth", alpha=0.8, lw=1)
        axes[0].plot(pred_p50, "r-", label="estimate", alpha=0.8, lw=1)
        axes[0].fill_between(
            torch.arange(0, T1), pred_p10, pred_p90, color="red", alpha=0.3
        )
        axes[0].legend()
        axes[0].set_title("response")
    else:
        axes[idx].plot(betas[:T1, idx - 1], "k-", label="truth", alpha=0.8, lw=1)
        axes[idx].plot(coef_p50[:, idx - 1], "r-", label="estimate", alpha=0.8, lw=1)
        if idx >= 2:
            axes[idx].plot(
                time_points, priors[:, idx - 2], "o", color="blue", alpha=0.8, lw=1
            )
        axes[idx].fill_between(
            torch.arange(0, T1),
            coef_p10[:, idx - 1],
            coef_p90[:, idx - 1],
            color="red",
            alpha=0.3,
        )
        axes[idx].set_title("coef_{}".format(idx - 1))
plt.tight_layout()
# %%
# ---predict---
pyro.set_rng_seed(1)
samples2 = forecaster2(y[:T1], covariates, num_samples=1000)
p10, p50, p90 = quantile(samples2, (0.1, 0.5, 0.9)).squeeze(-1)
crps = eval_crps(samples2, y[T1:])
print(samples2.shape, p10.shape)

plt.figure(figsize=(20, 5))
plt.fill_between(torch.arange(T1, T2), p10, p90, color="red", alpha=0.3)
plt.plot(torch.arange(T1, T2), p50, "r-", label="forecast", alpha=0.8)
plt.plot(np.arange(T1 - 200, T2), y[T1 - 200 : T2], "k-", label="truth", alpha=0.5)
plt.title("Response against time (CRPS = {:0.3g})".format(crps))
plt.axvline(T1, color="b", linestyle="--")
plt.legend(loc="best")
# %%
