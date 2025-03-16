# %%
from ucimlrepo import fetch_ucirepo
import polars as pl
import numpy as np
import seaborn as sns
import util
import matplotlib.pyplot as plt

# %%
# import iris dataset from https://archive.ics.uci.edu/dataset/53/iris

iris = fetch_ucirepo(id=53)
# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets
y.columns = ["species"]
# metadata
print(iris.metadata)

# variable information
print(iris.variables)

# %%
# read with polars
q = (
    pl.from_pandas(X)
    .lazy()
    .with_row_count("idx")
    .join(pl.from_pandas(y).lazy().with_row_count("idx"), on="idx")
    .drop("idx")
    .select(["sepal length", "petal length", "species"])
    .filter(
        (pl.col("species") == "Iris-setosa") | (pl.col("species") == "Iris-versicolor")
    )
    .with_columns(
        pl.when(pl.col("species") == "Iris-setosa")
        .then(0)
        .otherwise(1)
        .alias("species")
    )
)
print(q.explain())
df = q.collect()
X_train = df["sepal length", "petal length"].to_numpy()
y_train = df["species"].to_numpy()
# standardization
X_train_std = util.standardize(X_train)
# %%
# plot data with seaborn
sns.set_theme()
sns.relplot(data=df, x="sepal length", y="petal length", style="species", hue="species")
# %%
fig, axes = plt.subplots(1, 2)

sns.set_theme()
sns.histplot(X_train, bins=20, stat="probability", kde=True, ax=axes[0])
sns.histplot(X_train_std, bins=20, stat="probability", kde=True, ax=axes[1])


# %%
# train

ppn = util.Perceptron_Classifier(eta=0.1, n_iter=10)
ppn.fit(X_train, y_train)
# %%
# plot error
ppn.plot_losses()


# %%
util.plot_decision_regions(X_train, y_train, ppn)
# %%
ada1 = util.ADALINE_Classifier(n_iter=15, eta=0.01).fit(X_train, y_train)
ada2 = util.ADALINE_Classifier(n_iter=15, eta=0.0001).fit(X_train, y_train)
# plot error
ada1.plot_losses()
ada2.plot_losses()
# %%
# plot regions of ada2
util.plot_decision_regions(X_train, y_train, ada1)


# %%
ada_gd = util.ADALINE_Classifier(eta=0.5, n_iter=20)
ada_gd.fit(X_train_std, y_train)
# %%
ada_gd.plot_losses()

# %%
util.plot_decision_regions(X_train_std, y_train, ada_gd)

# %%
print(ada_gd.losses_)
# %%
# SGD ver.
ada_sgd = util.ADALINE_Classifier_SGD(n_iter=15, eta=0.01)
ada_sgd.fit(X_train_std, y_train)

# %%
ada_sgd.plot_losses()
# %%
util.plot_decision_regions(X_train_std, y_train, ada_sgd)
# %%
# ada_sgd.partial_fit(X_train_std, y_train)
