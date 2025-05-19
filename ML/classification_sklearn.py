# %%
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import util
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# %%
# sklearn
X, y = util.Iris()
X_train, X_test, y_train, y_test = util.split(X, y, test_size=0.3, stratify=y)
# %%
# これスニペットにする
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# %%
# random_state is for the shuffle at each epoch
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
