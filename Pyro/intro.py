# %%
import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import time

# %%
smoke_test = "CI" in os.environ
assert pyro.__version__.startswith("1.9.1")

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Set matplotlib settings
plt.style.use("default")
# %%
# ---example data---
data = pd.read_csv("rugged_data.csv", encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
# %%
# ---visualize rgdppc_2000---
plt.hist(df["rgdppc_2000"], bins=30)
plt.title("before log transform")
plt.show()
# %%
# ---log transform---
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
plt.hist(df["rgdppc_2000"], bins=30)
plt.title("after log transform")
plt.show()
# %%
# ---to tensor---
train = torch.tensor(df.values, dtype=torch.float)
is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
# %%
# ---visualize---
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]
sns.scatterplot(
    x=non_african_nations["rugged"], y=non_african_nations["rgdppc_2000"], ax=ax[0]
)
ax[0].set(
    xlabel="Terrain Ruggedness Index",
    ylabel="log GDP (2000)",
    title="Non African Nations",
)
sns.scatterplot(x=african_nations["rugged"], y=african_nations["rgdppc_2000"], ax=ax[1])
ax[1].set(
    xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations"
)
# %%
# ---define model(frequentist)---
"""
mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
"""
# equivalent to
# def simpler_model(is_cont_africa, ruggedness): ...
# conditioned_model = pyro.condition(simpler_model, data={"obs": log_gdp})


def simple_model(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.param("a", lambda: torch.randn(()))
    b_a = pyro.param("bA", lambda: torch.randn(()))
    b_r = pyro.param("bR", lambda: torch.randn(()))
    b_ar = pyro.param("bAR", lambda: torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)

    mean = (
        a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    )

    with pyro.plate("data", len(ruggedness)):
        # in this case, it will always return log_gdp
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)


pyro.render_model(
    simple_model,
    model_args=(is_cont_africa, ruggedness, log_gdp),
    render_distributions=True,
    render_params=True,
)


# %%
# ---bayesian model---
def model(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.sample("a", dist.Normal(0.0, 10.0))
    b_a = pyro.sample("bA", dist.Normal(0.0, 1.0))
    b_r = pyro.sample("bR", dist.Normal(0.0, 1.0))
    b_ar = pyro.sample("bAR", dist.Normal(0.0, 1.0))
    sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))

    mean = (
        a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    )

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)


pyro.render_model(
    model,
    model_args=(is_cont_africa, ruggedness, log_gdp),
    render_distributions=True,
    render_params=True,
)


# %%
# ---variational distribution---
# equivalent to
# auto_guide = pyro.infer.autoguide.AutoNormal(model)
def custom_guide(is_cont_africa, ruggedness, log_gdp=None):
    a_loc = pyro.param("a_loc", lambda: torch.tensor(0.0))
    a_scale = pyro.param(
        "a_scale", lambda: torch.tensor(1.0), constraint=constraints.positive
    )
    sigma_loc = pyro.param("sigma_loc", lambda: torch.tensor(0.0))
    weights_loc = pyro.param("weights_loc", lambda: torch.randn(3))
    weights_scale = pyro.param(
        "weights_scale", lambda: torch.ones(3), constraint=constraints.positive
    )
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample(
        "sigma", dist.LogNormal(sigma_loc, torch.tensor(0.05))
    )  # fixed scale for simplicity
    return {"a": a, "b_a": b_a, "b_r": b_r, "b_ar": b_ar, "sigma": sigma}


pyro.render_model(
    custom_guide,
    model_args=(is_cont_africa, ruggedness, log_gdp),
    render_distributions=True,
    render_params=True,
)
# %%
# ---compute---
pyro.clear_param_store()

# These should be reset each training loop.
auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
"""
 Sometimes models and guides are sensitive to learning rate,
 and the first thing to try is decreasing learning rate and increasing number of steps.
 This is especially important in models and guides with deep neural nets.
 We recommend starting with a lower learning rate and gradually increasing,
 avoiding learning rates that are too fast, where inference can diverge or result in NANs
"""
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []
start = time.time()
for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step(is_cont_africa, ruggedness, log_gdp)
    losses.append(loss)
    if step % 100 == 0:
        logging.info("Elbo loss: {}".format(loss))
end = time.time()
print("time", end - start)
plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
plt.show()
# %%
# --- best param---
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())
# %%
# ---visualize sample---
# sample from posterior
pyro.set_rng_seed(1)
with pyro.plate("samples", 800, dim=-1):
    samples = auto_guide(is_cont_africa, ruggedness)

gamma_within_africa = samples["bR"] + samples["bAR"]
gamma_outside_africa = samples["bR"]

fig = plt.figure(figsize=(10, 6))
sns.histplot(
    gamma_within_africa.detach().cpu().numpy(),
    kde=True,
    stat="density",
    label="African nations",
)
sns.histplot(
    gamma_outside_africa.detach().cpu().numpy(),
    kde=True,
    stat="density",
    label="Non-African nations",
    color="orange",
)
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness")
plt.xlabel("Slope of regression line")
plt.legend()
plt.show()
# %%
# ---visualize posterior sampling---
# sample
# predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=800)
predictive = pyro.infer.Predictive(model, posterior_samples=samples)
svi_samples = predictive(is_cont_africa, ruggedness, log_gdp=None)
svi_gdp = svi_samples["obs"]
# visualize
predictions = pd.DataFrame(
    {
        "cont_africa": is_cont_africa,
        "rugged": ruggedness,
        "y_mean": svi_gdp.mean(0).detach().cpu().numpy(),
        "y_perc_5": svi_gdp.kthvalue(int(len(svi_gdp) * 0.05), dim=0)[0]
        .detach()
        .cpu()
        .numpy(),
        "y_perc_95": svi_gdp.kthvalue(int(len(svi_gdp) * 0.95), dim=0)[0]
        .detach()
        .cpu()
        .numpy(),
        "true_gdp": log_gdp,
    }
)
african_nations = predictions[predictions["cont_africa"] == 1].sort_values(
    by=["rugged"]
)
non_african_nations = predictions[predictions["cont_africa"] == 0].sort_values(
    by=["rugged"]
)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
ax[0].fill_between(
    non_african_nations["rugged"],
    non_african_nations["y_perc_5"],
    non_african_nations["y_perc_95"],
    alpha=0.5,
)
ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
ax[0].set(
    xlabel="Terrain Ruggedness Index",
    ylabel="log GDP (2000)",
    title="Non African Nations",
)

ax[1].plot(african_nations["rugged"], african_nations["y_mean"])
ax[1].fill_between(
    african_nations["rugged"],
    african_nations["y_perc_5"],
    african_nations["y_perc_95"],
    alpha=0.5,
)
ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
ax[1].set(
    xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations"
)
plt.show()
# %%
# ---include covariation in model---
mvn_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
pyro.render_model(
    mvn_guide, model_args=(is_cont_africa, ruggedness, log_gdp), render_params=True
)
# %%
# ---train and sample from posterior---
pyro.set_rng_seed(1)
pyro.clear_param_store()
mvn_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
svi = pyro.infer.SVI(
    model, mvn_guide, pyro.optim.Adam({"lr": 0.02}), pyro.infer.Trace_ELBO()
)

losses = []
for step in range(1000 if not smoke_test else 2):
    loss = svi.step(is_cont_africa, ruggedness, log_gdp)
    losses.append(loss)
    if step % 100 == 0:
        logging.info("Elbo loss: {}".format(loss))

plt.figure(figsize=(5, 2))
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")

with pyro.plate("samples", 800, dim=-1):
    mvn_samples = mvn_guide(is_cont_africa, ruggedness)

mvn_gamma_within_africa = mvn_samples["bR"] + mvn_samples["bAR"]
mvn_gamma_outside_africa = mvn_samples["bR"]

# Interface note: reuse guide samples for prediction by passing them to Predictive
# via the posterior_samples keyword argument instead of passing the guide as above
assert "obs" not in mvn_samples
mvn_predictive = pyro.infer.Predictive(model, posterior_samples=mvn_samples)
mvn_predictive_samples = mvn_predictive(is_cont_africa, ruggedness, log_gdp=None)

mvn_gdp = mvn_predictive_samples["obs"]

# %%
# ---plot density---
svi_samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}
svi_mvn_samples = {k: v.detach().cpu().numpy() for k, v in mvn_samples.items()}
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-sections of the Posterior Distribution", fontsize=16)

# 左側のプロット
sns.kdeplot(
    x=svi_samples["bA"], y=svi_samples["bR"], ax=axs[0], bw_adjust=4, color="blue"
)
sns.kdeplot(
    x=svi_mvn_samples["bA"],
    y=svi_mvn_samples["bR"],
    ax=axs[0],
    fill=True,
    bw_adjust=4,
    color="red",
)
axs[0].set(xlabel="bA", ylabel="bR", xlim=(-2.8, -0.9), ylim=(-0.6, 0.2))

# 右側のプロット
sns.kdeplot(
    x=svi_samples["bR"], y=svi_samples["bAR"], ax=axs[1], bw_adjust=4, color="blue"
)
sns.kdeplot(
    x=svi_mvn_samples["bR"],
    y=svi_mvn_samples["bAR"],
    ax=axs[1],
    fill=True,
    bw_adjust=4,
    color="red",
)
axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.55, 0.2), ylim=(-0.15, 0.85))

# 凡例のためのダミープロットを作成する場合
for label, color in zip(
    ["SVI (Diagonal Normal)", "SVI (Multivariate Normal)"], ["blue", "red"]
):
    axs[0].plot([], [], label=label, color=color)


fig.legend(loc="upper right")
plt.show()
# %%
# ---cut density---
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness")

sns.histplot(
    gamma_within_africa.detach().cpu().numpy(),
    ax=axs[0],
    kde=True,
    stat="density",
    label="African nations",
)
sns.histplot(
    gamma_outside_africa.detach().cpu().numpy(),
    ax=axs[0],
    kde=True,
    stat="density",
    color="orange",
    label="Non-African nations",
)
axs[0].set(
    title="Mean field",
    xlabel="Slope of regression line",
    xlim=(-0.6, 0.6),
    ylim=(0, 11),
)

sns.histplot(
    mvn_gamma_within_africa.detach().cpu().numpy(),
    ax=axs[1],
    kde=True,
    stat="density",
    label="African nations",
)
sns.histplot(
    mvn_gamma_outside_africa.detach().cpu().numpy(),
    ax=axs[1],
    kde=True,
    stat="density",
    color="orange",
    label="Non-African nations",
)
axs[1].set(
    title="Full rank", xlabel="Slope of regression line", xlim=(-0.6, 0.6), ylim=(0, 11)
)

handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
# %%
# ---visualize CI---
mvn_predictions = pd.DataFrame(
    {
        "cont_africa": is_cont_africa,
        "rugged": ruggedness,
        "y_mean": mvn_gdp.mean(dim=0).detach().cpu().numpy(),
        "y_perc_5": mvn_gdp.kthvalue(int(len(mvn_gdp) * 0.05), dim=0)[0]
        .detach()
        .cpu()
        .numpy(),
        "y_perc_95": mvn_gdp.kthvalue(int(len(mvn_gdp) * 0.95), dim=0)[0]
        .detach()
        .cpu()
        .numpy(),
        "true_gdp": log_gdp,
    }
)
mvn_non_african_nations = mvn_predictions[
    mvn_predictions["cont_africa"] == 0
].sort_values(by=["rugged"])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
ax[0].fill_between(
    non_african_nations["rugged"],
    non_african_nations["y_perc_5"],
    non_african_nations["y_perc_95"],
    alpha=0.5,
)
ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
ax[0].set(
    xlabel="Terrain Ruggedness Index",
    ylabel="log GDP (2000)",
    title="Non African Nations: Mean-field",
)

ax[1].plot(mvn_non_african_nations["rugged"], mvn_non_african_nations["y_mean"])
ax[1].fill_between(
    mvn_non_african_nations["rugged"],
    mvn_non_african_nations["y_perc_5"],
    mvn_non_african_nations["y_perc_95"],
    alpha=0.5,
)
ax[1].plot(mvn_non_african_nations["rugged"], mvn_non_african_nations["true_gdp"], "o")
ax[1].set(
    xlabel="Terrain Ruggedness Index",
    ylabel="log GDP (2000)",
    title="Non-African Nations: Full rank",
)

# %%
