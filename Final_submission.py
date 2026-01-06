import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

def column(df):
    df.columns=[col.lower() for col in df.columns]
    return df

train=column(train)
test=column(test)

def lags(df):
    df=df.copy()
    for col in["open","high","low","close","volume"]:
        if col in df.columns:
            for i in range(21):
                df[f"{col}_lag_{i}"]=df[col].shift(i)
    return df

train=lags(train)
train=train.dropna().reset_index(drop=True)
test=lags(test)

def fengg(df):
    for i in [1,5,10,20]:
        if f"close_lag_{i}" in df.columns:
            df[f"logret_{i}"]=np.log(df["close_lag_0"]/df[f"close_lag_{i}"])
    for i in [5,10,20]:
        cols=[f"close_lag_{j}" for j in range(i) if f"close_lag_{j}" in df.columns]
        if cols:
            ma=df[cols].mean(axis=1)
            df[f"price_ma_{i}"]=df["close_lag_0"]/ma
    cols=[f"close_lag_{i}" for i in range(5) if f"close_lag_{i}" in df.columns]
    if cols:
        df["vol5"]=df[cols].std(axis=1)
    return df

train=fengg(train)
test=fengg(test)
train["y"]=np.log(train["target"]/train["close_lag_0"])
features=[]
for i in range(10):
    if f"close_lag_{i}" in train.columns:
        features.append(f"close_lag_{i}")
for i in [1,5,10,20]:
    if f"logret_{i}" in train.columns:
        features.append(f"logret_{i}")
for i in [5,10,20]:
    if f"price_ma_{i}" in train.columns:
        features.append(f"price_ma_{i}")
if "vol5" in train.columns:
    features.append("vol5")

X=train[features]
y=train["y"]
X_test=test[features]

gb=GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

et=ExtraTreesRegressor(
    n_estimators=400,
    max_depth=7,
    min_samples_leaf=10,
    random_state=42
)

model=VotingRegressor(
    [("gb",gb),("et",et)],
    weights=[2,1]
)

model.fit(X,y)


log_preds=model.predict(X_test)
preds=test["close_lag_0"]*np.exp(log_preds)


submission=pd.DataFrame({
    "ID":test["id"],
    "Target":preds
})

submission.to_csv("submission15.csv",index=False)