import os
import pickle
import random
import re
import subprocess
import sys
import time

import mlpack
import numpy as np
import pandas as pd
import psutil
import pyperclip
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# Usage: python runner.py <dataset>

# data set hardcoded information
file_map = {
    "iris": "iris.data",
    "chess": "kr-vs-kp.data",
    "tictactoe": "tic-tac-toe.data",
    "abalone": "abalone.data",
    "car": "car.data",
    "connect-4": "connect-4.data",
    "nursery": "nursery.data",
    "gamma": "magic04.data",
    "spam": "spambase.data",
    "poker": "poker-hand-testing.data",
}

contMap = {
    "iris":      [True] * 4,
    "chess":     [False] * 36,
    "tictactoe": [False] * 9,
    "abalone":   [False] + [True] * 7,
    "car":       [False] * 6,
    "connect-4": [False] * 42,
    "nursery":   [False] * 8,
    "gamma":     [True] * 10,
    "spam":     [True] * 57,
    "poker":     [False] * 10,
}

# Read user arguments
if len(sys.argv) < 2 or sys.argv[1] not in file_map:
    print("Please provide a valid dataset:", ", ".join(file_map.keys()))
    sys.exit(1)

dataset = sys.argv[1]
filename = file_map[dataset]
is_cont = contMap[dataset]

# Read input file and separate random 70% for training and rest for validation
with open(filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

random.shuffle(lines)
split_idx = int(len(lines) * 0.7)

with open("train.tmp", "w") as f:
    f.write("\n".join(lines[:split_idx]) + "\n")
with open("validate.tmp", "w") as f:
    f.write("\n".join(lines[split_idx:]) + "\n")

formatted_rows = []
print(f"--- Benchmarking '{dataset}' ---")

# Custom C++ Implementation
for criteria in ["GAIN", "GAIN_RATIO", "GINI"]:
    result = subprocess.run(
        ["./main", criteria, dataset, "train.tmp", "validate.tmp"],
        capture_output=True,
        text=True,
    )
    matches = re.findall(
        r"Time:\s*([-\d.]+)\s*ms\s*\|\s*Peak:\s*([-\d.]+)\s*KB\s*\|\s*Net:\s*([-\d.]+)\s*KB",
        result.stdout,
    )
    stats = re.search(r"Accuracy:\s*([-\d.]+)\s*\|\s*NULL Count:\s*([-\d.]+)\s", result.stdout)

    if len(matches) == 2 and stats:
        row_vals = [criteria]
        for time_str, peak_str, net_str in matches:
            row_vals.extend([float(time_str), float(net_str), float(peak_str)])
        row_vals.append(float(stats.group(1)))
        formatted_rows.append("\t".join(map(str, row_vals)))

# Preprocess data for reference algorithm to understand
rows = [line.split(",") for line in lines]
X_raw = [r[:-1] for r in rows]
y_raw = [r[-1] for r in rows]

y_all = LabelEncoder().fit_transform(y_raw)
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

cont_cols = [i for i, flag in enumerate(is_cont) if flag]
cat_cols = [i for i, flag in enumerate(is_cont) if not flag]

# scikit-learn DT expects values that are contiuous, we use one-hot encoding to deal with discrete values
df_raw = pd.DataFrame(X_raw)
for c in cont_cols:
    df_raw[c] = df_raw[c].astype(float)

ohe_prep = ColumnTransformer(
    transformers=[
        ("num", "passthrough", cont_cols),
        ("cat", OneHotEncoder(sparse_output=False), cat_cols),
    ]
)
X_ohe = ohe_prep.fit_transform(df_raw)
X_ohe_train, X_ohe_test = X_ohe[:split_idx], X_ohe[split_idx:]

# Tell the mlpack's dt which values are contiuous/discrete, make copy to not change raw data
df_mlp = df_raw.copy()
for c in cat_cols:
    df_mlp[c] = df_mlp[c].astype("category")
X_mlp_train, X_mlp_test = df_mlp.iloc[:split_idx], df_mlp.iloc[split_idx:]

# For naive bayes, split the data between contiuous and discrete and call two separate
# algorithms (gaussain for contiuous, categorical for discrete)
X_cont_train = df_raw[cont_cols].iloc[:split_idx].to_numpy(dtype=float) if cont_cols else None
X_cont_test = df_raw[cont_cols].iloc[split_idx:].to_numpy(dtype=float) if cont_cols else None

if cat_cols:
    ord_enc = OrdinalEncoder()
    X_cat_all = ord_enc.fit_transform(df_raw[cat_cols]).astype(int)
    X_cat_train, X_cat_test = X_cat_all[:split_idx], X_cat_all[split_idx:]
    cat_sizes = [len(cats) for cats in ord_enc.categories_]
else:
    X_cat_train = X_cat_test = cat_sizes = None

process = psutil.Process(os.getpid())

# Record statistics (Memory, time) for each reference algorithm when run
def record_row(name, train_fn, pred_fn):
    mem_b = process.memory_info().rss
    t0 = time.perf_counter()
    model = train_fn()
    train_ms = round((time.perf_counter() - t0) * 1000, 2)
    mem_a = process.memory_info().rss

    train_peak = round(max(0.0, (mem_a - mem_b) / 1024.0), 2)
    train_net = round(len(pickle.dumps(model)) / 1024.0, 2)

    mem_b = process.memory_info().rss
    t0 = time.perf_counter()
    y_pred = pred_fn(model)
    pred_ms = round((time.perf_counter() - t0) * 1000, 2)
    mem_a = process.memory_info().rss

    pred_peak = round(max(0.0, (mem_a - mem_b) / 1024.0), 2)
    acc = round((np.sum(y_pred == y_test) / len(y_test)) * 100, 2)

    row = [name, train_ms, train_net, train_peak, pred_ms, 0, pred_peak, acc]
    formatted_rows.append("\t".join(map(str, row)))

# Run SK algorithm (GINI and Entropy) and measure results
for label, crit in [
    ("SK-DT (Entropy)", "entropy"),
    ("SK-DT (Gini)", "gini"),
]:
    clf = DecisionTreeClassifier(criterion=crit)
    record_row(
        label,
        lambda c=clf: c.fit(X_ohe_train, y_train),
        lambda c: c.predict(X_ohe_test),
    )

# run mlpack's dt algorithm and measure results
mlp_clf = mlpack.decision_tree.DecisionTree(
    minimum_leaf_size=1
)

def train_mlp():
    mlp_clf.fit(training=X_mlp_train, labels=y_train)
    return mlp_clf

record_row(
    "MLP-DT",
    train_mlp,
    lambda m: np.asarray(m.predict(test=X_mlp_test)).ravel(),
)

# run SK naive Bayesian algorithm and measure results, we use a method
# where continuous values are fed into GaussianNB while discrete are fed into CategorialNB
# multiple the results and divded by the class prior probability and we get the answer
def train_nb():
    gnb = GaussianNB().fit(X_cont_train, y_train) if cont_cols else None
    cnb = CategoricalNB(min_categories=cat_sizes).fit(X_cat_train, y_train) if cat_cols else None
    return (gnb, cnb)

def pred_nb(models):
    gnb, cnb = models
    if gnb and cnb:
        proba = (gnb.predict_proba(X_cont_test) * cnb.predict_proba(X_cat_test)) / gnb.class_prior_
        return np.argmax(proba, axis=1)
    return gnb.predict(X_cont_test) if gnb else cnb.predict(X_cat_test)

record_row("SK-NB", train_nb, pred_nb)

# Copy the results into clipboard for easy pasting into google sheets
pyperclip.copy("\n".join(formatted_rows))
print(f"Copied {len(formatted_rows)} rows for '{dataset}' straight to clipboard!")
