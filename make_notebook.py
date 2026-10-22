"""Assemble Bank_Marketing_ML_Labs.ipynb from cell sources below."""
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Bank Telemarketing Success Prediction — ML Lab Implementations
**Dataset:** UCI Bank Marketing, `bank-full.csv` (45,211 calls, 17 columns), Moro, Cortez & Rita (2014).
**Task:** predict whether a client subscribes to a term deposit (`y = yes`) **before** the call is made.

| Section | Lab assignment | What is done here |
|---|---|---|
| 1 | Lab 1 — Cleaning & EDA | missing values, outliers, encoding, distributions, correlation, feature selection, PCA |
| 2 | Lab 3 — Regularization | no penalty vs L1 / L2 / ElasticNet logistic regression |
| 3 | Lab 4 — Classification & evaluation | 2014 baselines vs Random Forest, LightGBM, XGBoost, CatBoost and ensembles; imbalance handling; Optuna tuning |
| 4 | Lab 5 — RFM + K-Means | recency / frequency / monetary features, elbow & silhouette, cluster strategies |
| 5 | Lab 7 — Genetic Algorithm feature selection | GA over feature groups vs all / L1 / mutual-information subsets |
| 6 | Explainability | TreeSHAP values of the final model |

Labs 2 (house-price regression) and 6 (traffic-light RL) need different data/environments and are not applicable to this dataset.
""")

code(r"""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score,
                                     cross_val_predict, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, precision_recall_curve,
                             brier_score_loss, silhouette_score, matthews_corrcoef,
                             accuracy_score, balanced_accuracy_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
FIG, RES = "figures/", "results/"
import os; os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
RESULTS = {}
def save(name): plt.tight_layout(); plt.savefig(FIG + name, bbox_inches="tight"); plt.show()
""")

# ------------------------------------------------------------ LAB 1
md("# 1. Lab 1 — Data cleaning and exploratory data analysis")
code(r"""
df = pd.read_csv("bank-full.csv", sep=";")
print(df.shape); display(df.head())
df.info()
print("\nDuplicate rows:", df.duplicated().sum())
print(df["y"].value_counts(normalize=True).round(4))
RESULTS["dataset"] = {"rows": len(df), "cols": df.shape[1],
                      "positive_rate": float((df.y == "yes").mean()),
                      "imbalance_ratio": float((df.y == "no").sum() / (df.y == "yes").sum())}
# Fix the 80/20 split now, so every learned statistic (caps, checks) can use training rows only.
IDX_TR, IDX_TE = train_test_split(df.index, test_size=.2, stratify=df.y, random_state=SEED)
print("train rows:", len(IDX_TR), "| test rows:", len(IDX_TE))
""")

md("## 1.1 Missing values\nThe file has no empty cells, but missingness is hidden in **sentinels**: the string `unknown` and `pdays = -1` (never contacted).")
code(r"""
print("Literal NaNs:", int(df.isna().sum().sum()))
cat_cols = [c for c in df.columns if df[c].dtype == object and c != "y"]
num_cols = [c for c in df.columns if df[c].dtype != object]
unk = pd.DataFrame({
    "unknown_count": [(df[c] == "unknown").sum() for c in cat_cols],
    "unknown_pct": [(df[c] == "unknown").mean() * 100 for c in cat_cols],
    "conv_rate_if_unknown": [(df.loc[df[c] == "unknown", "y"] == "yes").mean() * 100 for c in cat_cols],
    "conv_rate_if_known": [(df.loc[df[c] != "unknown", "y"] == "yes").mean() * 100 for c in cat_cols]},
    index=cat_cols).query("unknown_count > 0").round(2)
display(unk)
print("pdays == -1 (never contacted):", (df.pdays == -1).sum(), f"({(df.pdays==-1).mean()*100:.1f}%)")
RESULTS["unknown"] = unk.to_dict()
""")
md(r"""
**Treatment decision.** `unknown` converts at a very different rate from known values (e.g. `contact = unknown` ≈ 4% vs ≈ 15%), so it is **not missing at random**. Imputing it with the mode would destroy that signal.
* `unknown` is kept as its own category, and an explicit indicator (`unk_job`, `unk_education`, `unk_contact`, `unk_poutcome`) is added.
* `pdays = -1` is recoded to `NaN` with a `never_contacted` flag, because keeping -1 would tell a model "contacted yesterday".
* `duration` (call length) is **removed** from the predictive feature set. It is only known *after* the call, so it cannot be used to decide whom to call (data leakage). It is kept aside only to measure the leakage.
""")
code(r"""
# Mode imputation vs keeping 'unknown' — quick check with the same simple model (training rows only)
def quick_auc(frame):
    X = pd.get_dummies(frame.drop(columns=["y", "duration"]), drop_first=True).astype(float)
    return cross_val_score(Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]), X,
                           (frame.y == "yes").astype(int), cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc").mean()
imp = df.loc[IDX_TR].copy()
for c in ["job", "education", "contact", "poutcome"]:
    imp[c] = imp[c].replace("unknown", imp.loc[imp[c] != "unknown", c].mode()[0])
mv = {"keep_unknown_as_category": quick_auc(df.loc[IDX_TR]), "mode_imputation": quick_auc(imp)}
print({k: round(v, 4) for k, v in mv.items()}); RESULTS["missing_value_treatment_auc"] = mv
""")

md("## 1.2 Outlier detection and treatment (IQR rule)")
code(r"""
out_cols = ["age", "balance", "campaign", "previous", "duration"]
rows = []
for c in out_cols:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows.append([c, lo, hi, int(((df[c] < lo) | (df[c] > hi)).sum()), ((df[c] < lo) | (df[c] > hi)).mean() * 100])
outl = pd.DataFrame(rows, columns=["feature", "lower", "upper", "n_outliers", "pct"]).round(2)
display(outl); RESULTS["outliers"] = outl.to_dict("records")

fig, ax = plt.subplots(1, 4, figsize=(13, 3))
for a, c in zip(ax, ["age", "balance", "campaign", "previous"]):
    sns.boxplot(x=df[c], ax=a, color="#7fa7d6", fliersize=1.5); a.set_title(c)
save("lab1_outliers_boxplots.png")
""")
md(r"""
**Treatment decision.** These outliers are real customers (a client really can hold a large balance or be called 40 times), so rows are **not deleted**. Instead, extreme values are **capped (winsorised) at the 1st and 99th percentiles of the training rows**. `balance` also gets a signed-log transform. Tree models are insensitive to outliers anyway; capping mainly protects the linear and neural models.
""")
code(r"""
CAPS = {c: tuple(df.loc[IDX_TR, c].quantile([.01, .99])) for c in ["age", "balance", "campaign", "previous"]}   # training rows only
print({k: tuple(round(x, 1) for x in v) for k, v in CAPS.items()})
""")

md("## 1.3 Distributions")
code(r"""
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
for a, c in zip(ax[0], ["age", "balance", "campaign"]):
    sns.histplot(data=df, x=c, hue="y", bins=40, stat="density", common_norm=False, ax=a,
                 palette={"no": "#9aa5b1", "yes": "#d1603d"})
    if c != "age": a.set_xlim(df[c].quantile(.005), df[c].quantile(.99))
for a, c in zip(ax[1], ["job", "month", "poutcome"]):
    r = df.groupby(c)["y"].apply(lambda s: (s == "yes").mean() * 100).sort_values()
    r.plot.barh(ax=a, color="#2e75b6"); a.axvline(11.7, ls="--", c="k", lw=.8)
    a.set_xlabel("conversion %"); a.set_title(f"Conversion by {c}")
save("lab1_distributions.png")
mrate = df.groupby("month")["y"].apply(lambda s: (s == "yes").mean())
RESULTS["month_conv_min_max"] = [float(mrate.min()), float(mrate.max())]
""")

md("## 1.4 Feature engineering and converting categorical data to numerical")
code(r"""
MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
EDU = {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": np.nan}

def engineer(d, caps=CAPS):
    d = d.copy()
    for c, (lo, hi) in caps.items(): d[c] = d[c].clip(lo, hi)            # outlier capping
    for c in ["default", "housing", "loan"]: d[c] = (d[c] == "yes").astype(int)  # binary -> 0/1
    d["education_level"] = d["education"].map(EDU)                       # ordinal encoding
    for c in ["job", "education", "contact", "poutcome"]:                 # informative missingness
        d[f"unk_{c}"] = (d[c] == "unknown").astype(int)
    d["never_contacted"] = (d.pdays == -1).astype(int)                   # sentinel recoding
    d["days_since_contact"] = d.pdays.where(d.pdays >= 0)
    d["total_contacts"] = d.campaign + d.previous
    d["prior_success"] = (d.poutcome == "success").astype(int)
    d["balance_slog"] = np.sign(d.balance) * np.log1p(np.abs(d.balance))
    d["negative_balance"] = (d.balance < 0).astype(int)
    m = d.month.map({m: i for i, m in enumerate(MONTHS)})
    d["month_sin"], d["month_cos"] = np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)  # cyclical encoding
    d["day_sin"], d["day_cos"] = np.sin(2*np.pi*d.day/31), np.cos(2*np.pi*d.day/31)
    return d

D = engineer(df)
y = (D.pop("y") == "yes").astype(int).values
duration = D.pop("duration")
NOMINAL = ["job", "marital", "education", "contact", "month", "poutcome"]
NUMERIC = [c for c in D.columns if c not in NOMINAL]
print(len(NOMINAL), "nominal +", len(NUMERIC), "numeric columns:", NUMERIC)

def make_pre(scale=True):
    num = Pipeline([("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else []))
    return ColumnTransformer([("num", num, NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)])
Xall = make_pre().fit_transform(D)
_enc = make_pre().fit(D).named_transformers_["cat"]
feat_names = NUMERIC + list(_enc.get_feature_names_out(NOMINAL))
# Parent attribute of every encoded column (by position, not by name prefix: e.g. month_sin is NOT part of 'month').
FEAT_GROUP = NUMERIC + [g for g, cats in zip(NOMINAL, _enc.categories_) for _ in cats]
assert len(FEAT_GROUP) == len(feat_names) and len(set(feat_names)) == len(feat_names)
print("Final model matrix:", Xall.shape)
RESULTS["n_features_encoded"] = Xall.shape[1]
""")

md("## 1.5 Correlation analysis")
code(r"""
num_view = D[NUMERIC].copy(); num_view["y"] = y; num_view["duration (post-call)"] = duration.values
corr = num_view.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 6.5}, cbar_kws={"shrink": .7})
plt.title("Pearson correlation (numeric + engineered features)")
save("lab1_correlation_heatmap.png")
tc = corr["y"].drop("y").sort_values(key=abs, ascending=False)
display(tc.head(10).round(3)); RESULTS["top_corr_with_target"] = tc.head(10).round(4).to_dict()
hi_pairs = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(corr) for b in corr.columns[i+1:]
            if abs(corr.loc[a, b]) > .8 and "y" not in (a, b)]
print("Highly correlated pairs (|r|>0.8):", hi_pairs)
""")
md("Most numeric features have only weak linear correlation with the target. `duration` is the strongest (≈0.39), which is exactly the leakage problem: it measures the outcome of the call. `prior_success`, `never_contacted` and `days_since_contact` are the strongest **pre-call** signals. The highly correlated pairs are expected redundancies (for example `pdays` and `never_contacted`), which motivates feature selection.")

md("## 1.6 Feature selection — mutual information and chi-square")
code(r"""
mi = mutual_info_classif(Xall, y, random_state=SEED)
Xpos = Xall - Xall.min(axis=0)
ch = chi2(Xpos, y)[0]
fs = pd.DataFrame({"feature": feat_names, "mutual_info": mi, "chi2": ch}).sort_values("mutual_info", ascending=False)
display(fs.head(15).round(4))
fs.head(20).plot.barh(x="feature", y="mutual_info", figsize=(7, 6), legend=False, color="#2e75b6").invert_yaxis()
plt.title("Top-20 features by mutual information"); save("lab1_mutual_info.png")
RESULTS["mi_top15"] = fs.head(15)["feature"].tolist()
""")

md("## 1.7 Principal Component Analysis")
code(r"""
pca = PCA().fit(Xall)
cum = np.cumsum(pca.explained_variance_ratio_)
k90, k95 = int(np.argmax(cum >= .90) + 1), int(np.argmax(cum >= .95) + 1)
Z = PCA(2, random_state=SEED).fit_transform(Xall)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(range(1, len(cum)+1), cum, marker=".", c="#2e75b6")
for k, l in [(k90, "90%"), (k95, "95%")]: ax[0].axvline(k, ls="--", c="grey", lw=.8); ax[0].text(k+.5, .5, f"{l}: {k} PCs")
ax[0].set(xlabel="number of components", ylabel="cumulative explained variance", title="PCA scree (cumulative)")
idx = np.random.RandomState(SEED).choice(len(Z), 6000, replace=False)
ax[1].scatter(Z[idx, 0], Z[idx, 1], c=np.where(y[idx] == 1, "#d1603d", "#9aa5b1"), s=4, alpha=.5)
ax[1].set(title="First two principal components (orange = subscribed)", xlabel="PC1", ylabel="PC2")
save("lab1_pca.png")

cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
pca_res = {}   # CV on training rows only; scaling and PCA are fitted inside each fold
for name, steps in [("all features", []), (f"PCA {k90} comps (90% var)", [("pca", PCA(k90, random_state=SEED))])]:
    pipe = Pipeline([("pre", make_pre())] + steps + [("lr", LogisticRegression(max_iter=2000))])
    pca_res[name] = cross_val_score(pipe, D.loc[IDX_TR], y[IDX_TR], cv=cv5, scoring="roc_auc").mean()
print(f"{Xall.shape[1]} features -> 90% variance needs {k90} PCs, 95% needs {k95}")
print({k: round(v, 4) for k, v in pca_res.items()})
RESULTS["pca"] = {"n_features": Xall.shape[1], "k90": k90, "k95": k95, "lr_auc": pca_res}
""")
md("PCA compresses the feature space, but the two classes overlap heavily in the first two components, so this problem is not linearly separable in a low-dimensional projection. Dropping to the 90%-variance components costs some AUC, because PCA keeps directions of high *variance*, not of high *predictive value*. For that reason PCA is used here for analysis, and supervised selection (Section 5, GA) is used for the final model.")

# ------------------------------------------------------------ Train/test split
md("# Train / test protocol\nEverything below uses one stratified 80/20 split. Model selection, tuning, threshold choice and GA search use **only the 80% training part** with cross-validation (5-fold; 3-fold inside Optuna tuning and the GA). Outlier caps and the two EDA model checks above also used training rows only. The 20% test set is touched once, at the end, for the reported numbers.")
code(r"""
D_tr, D_te, y_tr, y_te, dur_tr, dur_te = train_test_split(D, y, duration.values, test_size=.2,
                                                          stratify=y, random_state=SEED)
assert (D_tr.index == IDX_TR).all() and (D_te.index == IDX_TE).all()   # same split as used for the caps
print(D_tr.shape, D_te.shape, y_tr.mean().round(4), y_te.mean().round(4))

def evaluate(p, yt, thr=0.5):
    pred = (p >= thr).astype(int)
    order = np.argsort(-p); top = order[: int(.1 * len(p))]
    return {"roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
            "precision": precision_score(yt, pred, zero_division=0), "recall": recall_score(yt, pred),
            "f1": f1_score(yt, pred), "mcc": matthews_corrcoef(yt, pred), "brier": brier_score_loss(yt, p),
            "top10_capture": yt[top].sum() / yt.sum(), "threshold": thr,
            "accuracy": accuracy_score(yt, pred), "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "tp": int(((pred == 1) & (yt == 1)).sum()), "fp": int(((pred == 1) & (yt == 0)).sum()),
            "fn": int(((pred == 0) & (yt == 1)).sum()), "tn": int(((pred == 0) & (yt == 0)).sum())}
""")

# ------------------------------------------------------------ LAB 3
md("# 2. Lab 3 — Regularization (L1, L2, ElasticNet)")
code(r"""
reg_rows, coef_path = [], []
Cs = [0.001, 0.01, 0.1, 1, 10]
for pen in ["l1", "l2", "elasticnet"]:
    for C in Cs:
        clf = LogisticRegression(penalty=pen, C=C, solver="saga", l1_ratio=.5 if pen == "elasticnet" else None,
                                 max_iter=3000, tol=1e-3)
        pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
        cv = __import__("sklearn.model_selection", fromlist=["cross_validate"]).cross_validate(
            pipe, D_tr, y_tr, cv=cv5, scoring="roc_auc", return_train_score=True)
        pipe.fit(D_tr, y_tr); nz = int((np.abs(pipe["clf"].coef_) > 1e-6).sum())
        reg_rows.append([pen, C, cv["train_score"].mean(), cv["test_score"].mean(), nz])
none = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(penalty=None, max_iter=3000))])
cvn = __import__("sklearn.model_selection", fromlist=["cross_validate"]).cross_validate(
    none, D_tr, y_tr, cv=cv5, scoring="roc_auc", return_train_score=True)
reg_rows.append(["none", np.nan, cvn["train_score"].mean(), cvn["test_score"].mean(), Xall.shape[1]])
reg = pd.DataFrame(reg_rows, columns=["penalty", "C", "train_auc", "cv_auc", "nonzero_coefs"])
reg["gap"] = reg.train_auc - reg.cv_auc
display(reg.round(4)); RESULTS["regularization"] = reg.round(5).to_dict("records")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for pen, g in reg[reg.penalty != "none"].groupby("penalty"):
    ax[0].plot(g.C, g.cv_auc, marker="o", label=pen); ax[1].plot(g.C, g.nonzero_coefs, marker="o", label=pen)
ax[0].axhline(cvn["test_score"].mean(), ls="--", c="k", lw=.8, label="no penalty")
ax[0].set(xscale="log", xlabel="C (inverse regularisation strength)", ylabel="CV ROC-AUC", title="Validation AUC vs regularisation")
ax[1].set(xscale="log", xlabel="C", ylabel="non-zero coefficients", title="Sparsity: L1 removes features")
ax[0].legend(); ax[1].legend(); save("lab3_regularization.png")
best_reg = reg.sort_values("cv_auc", ascending=False).iloc[0]; print("Best:", dict(best_reg))
""")
md("The train–validation gap is small for every setting, so logistic regression on this data does not overfit badly. Its limit is **underfitting**: it is a linear model. Strong regularisation (small C) hurts AUC. L1 is useful because it drops many one-hot columns to exactly zero while keeping almost the same AUC, which makes it a free feature selector. This is used as a baseline for the GA in Section 5.")

# ------------------------------------------------------------ LAB 4
md("# 3. Lab 4 — Classification and evaluation\n## 3.1 The 2014 baselines vs modern ensembles (5-fold CV on the training set)")
code(r"""
def lgbm(**kw):
    base = dict(n_estimators=400, learning_rate=.05, num_leaves=31, subsample=.8, subsample_freq=1,
                colsample_bytree=.8, random_state=SEED, verbose=-1, n_jobs=2)
    base.update(kw); return lgb.LGBMClassifier(**base)

models = {
    "Logistic Regression (2014)": (make_pre(), LogisticRegression(C=float(best_reg.C) if best_reg.penalty!="none" else 1.0,
                                     penalty=best_reg.penalty if best_reg.penalty!="none" else None,
                                     solver="saga", l1_ratio=.5 if best_reg.penalty=="elasticnet" else None, max_iter=3000, tol=1e-3)),
    "Decision Tree (2014)": (make_pre(False), DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED)),
    "Neural Network / MLP (2014)": (make_pre(), MLPClassifier((64, 32), alpha=1e-2, early_stopping=True, max_iter=300, random_state=SEED)),
    "Linear SVM (2014)": (make_pre(), CalibratedClassifierCV(LinearSVC(C=.1, dual=False), cv=3)),
    "Random Forest": (make_pre(False), RandomForestClassifier(400, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
    "Hist Gradient Boosting": (make_pre(False), HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, random_state=SEED)),
    "LightGBM": (make_pre(False), lgbm()),
}
cmp_rows, OOF = [], {}
for name, (pre, clf) in models.items():
    t = time.time()
    p = cross_val_predict(Pipeline([("pre", pre), ("clf", clf)]), D_tr, y_tr, cv=cv5, method="predict_proba")[:, 1]
    OOF[name] = p; r = evaluate(p, y_tr); r.update(model=name, seconds=time.time() - t); cmp_rows.append(r)
    print(f"{name:30s} AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  recall@0.5 {r['recall']:.3f}  ({r['seconds']:.0f}s)")
cmp = pd.DataFrame(cmp_rows).set_index("model"); RESULTS["model_comparison_cv"] = cmp.round(5).reset_index().to_dict("records")
""")

md("## 3.2 Class imbalance handling (on LightGBM)")
code(r"""
imb_set = {
    "none": ImbPipeline([("pre", make_pre(False)), ("clf", lgbm())]),
    "class weights": ImbPipeline([("pre", make_pre(False)), ("clf", lgbm(class_weight="balanced"))]),
    "SMOTE (inside CV)": ImbPipeline([("pre", make_pre(False)), ("sm", SMOTE(random_state=SEED)), ("clf", lgbm())]),
    "random undersampling": ImbPipeline([("pre", make_pre(False)), ("rus", RandomUnderSampler(random_state=SEED)), ("clf", lgbm())]),
}
imb_rows = []
for name, pipe in imb_set.items():
    p = cross_val_predict(pipe, D_tr, y_tr, cv=cv5, method="predict_proba")[:, 1]
    r = evaluate(p, y_tr); r["strategy"] = name; imb_rows.append(r)
imb = pd.DataFrame(imb_rows).set_index("strategy")
display(imb[["roc_auc", "pr_auc", "precision", "recall", "f1"]].round(4))
RESULTS["imbalance_cv"] = imb.round(5).reset_index().to_dict("records")
""")
md("Rebalancing barely changes the ranking quality (ROC-AUC), but it moves the operating point a lot: recall at the default 0.5 threshold jumps with class weighting or undersampling (SMOTE inside the fold changes it little for LightGBM). The same effect can be obtained more cleanly by **tuning the decision threshold** on the unbalanced model, which also keeps the probabilities calibrated. The final model uses that approach.")

md("## 3.3 Building the improved model: Optuna-tuned LightGBM, XGBoost, CatBoost and ensembles\nThree gradient-boosting libraries are tuned or configured, then combined. The final model is chosen by **cross-validated PR-AUC on the training set**, never by the test set.")
code(r"""
import optuna, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
optuna.logging.set_verbosity(optuna.logging.WARNING)
cv3 = StratifiedKFold(3, shuffle=True, random_state=SEED)
Xtr_t = make_pre(False).fit(D_tr).transform(D_tr)

def xgbm(**kw):
    return xgb.XGBClassifier(**{**dict(tree_method="hist", random_state=SEED, n_jobs=2, eval_metric="logloss"), **kw})

def obj_lgb(t):
    p = dict(n_estimators=t.suggest_int("n_estimators", 200, 1200), learning_rate=t.suggest_float("learning_rate", .01, .1, log=True),
             num_leaves=t.suggest_int("num_leaves", 8, 96), min_child_samples=t.suggest_int("min_child_samples", 10, 200),
             colsample_bytree=t.suggest_float("colsample_bytree", .3, 1), subsample=t.suggest_float("subsample", .5, 1),
             reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
    return cross_val_score(lgbm(**p), Xtr_t, y_tr, cv=cv3, scoring="average_precision").mean()
def obj_xgb(t):
    p = dict(n_estimators=t.suggest_int("n_estimators", 200, 1200), learning_rate=t.suggest_float("learning_rate", .01, .1, log=True),
             max_depth=t.suggest_int("max_depth", 3, 9), min_child_weight=t.suggest_float("min_child_weight", 1, 20, log=True),
             colsample_bytree=t.suggest_float("colsample_bytree", .3, 1), subsample=t.suggest_float("subsample", .5, 1),
             reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True), gamma=t.suggest_float("gamma", 1e-3, 5, log=True))
    return cross_val_score(xgbm(**p), Xtr_t, y_tr, cv=cv3, scoring="average_precision").mean()

t = time.time()
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj_lgb, n_trials=40)
BEST_LGB = st.best_params; print(f"Optuna LightGBM ({time.time()-t:.0f}s):", BEST_LGB)
t = time.time()
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj_xgb, n_trials=40)
BEST_XGB = st.best_params; print(f"Optuna XGBoost ({time.time()-t:.0f}s):", BEST_XGB)
RESULTS["tuned_lgbm_params"] = BEST_LGB; RESULTS["tuned_xgb_params"] = BEST_XGB

class Blend(BaseEstimator, ClassifierMixin):
    # Equal-weight average of predicted probabilities of several fitted models.
    def __init__(self, members): self.members = members
    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]; self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.fitted_], axis=0); return np.column_stack([1 - p, p])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= .5).astype(int)

class CatBoostWrap(BaseEstimator, ClassifierMixin):
    # sklearn-compatible wrapper: CatBoost receives the nominal columns natively (no one-hot encoding).
    def __init__(self, iterations=1000, learning_rate=.04, depth=6, l2_leaf_reg=5):
        self.iterations, self.learning_rate, self.depth, self.l2_leaf_reg = iterations, learning_rate, depth, l2_leaf_reg
    def fit(self, X, y):
        self.model_ = CatBoostClassifier(iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
                                         l2_leaf_reg=self.l2_leaf_reg, random_seed=SEED, verbose=0, thread_count=2,
                                         cat_features=NOMINAL).fit(X, y)
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X): return self.model_.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= .5).astype(int)

CANDIDATES = {
    "Tuned LightGBM": lambda: Pipeline([("pre", make_pre(False)), ("clf", lgbm(**BEST_LGB))]),
    "Tuned XGBoost": lambda: Pipeline([("pre", make_pre(False)), ("clf", xgbm(**BEST_XGB))]),
    "CatBoost": lambda: CatBoostWrap(),
    "Stacked ensemble (LGB+HGB+LR)": lambda: Pipeline([("pre", make_pre()), ("stack", StackingClassifier(
        estimators=[("lgbm", lgbm(**BEST_LGB)),
                    ("hgb", HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, random_state=SEED)),
                    ("lr", LogisticRegression(max_iter=3000))],
        final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method="predict_proba", n_jobs=1))]),
}
for name, mk in CANDIDATES.items():
    t = time.time(); p = cross_val_predict(mk(), D_tr, y_tr, cv=cv5, method="predict_proba")[:, 1]
    OOF[name] = p; r = evaluate(p, y_tr); r["model"] = name; cmp.loc[name] = pd.Series(r)
    print(f"{name:32s} CV AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  ({time.time()-t:.0f}s)")

BLENDS = {"Blend LightGBM+CatBoost": ["Tuned LightGBM", "CatBoost"],
          "Blend XGBoost+CatBoost": ["Tuned XGBoost", "CatBoost"],
          "Blend LightGBM+XGBoost+CatBoost": ["Tuned LightGBM", "Tuned XGBoost", "CatBoost"]}
for name, parts in BLENDS.items():
    CANDIDATES[name] = (lambda parts=parts: Blend([CANDIDATES[q]() for q in parts]))
    p = np.mean([OOF[q] for q in parts], axis=0); OOF[name] = p
    r = evaluate(p, y_tr); r["model"] = name; cmp.loc[name] = pd.Series(r)
    print(f"{name:32s} CV AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}")

cand = cmp.loc[list(CANDIDATES)].sort_values(["pr_auc", "roc_auc"], ascending=False)
FINAL_NAME = cand.index[0]
print("\nSelected by CV PR-AUC ->", FINAL_NAME)
RESULTS["candidates_cv"] = cand[["roc_auc", "pr_auc"]].round(5).reset_index().to_dict("records")
RESULTS["final_model_name"] = FINAL_NAME
OOF["Final model (ours)"] = OOF[FINAL_NAME]
""")

md("## 3.4 Choosing the decision threshold (on training OOF predictions only)")
code(r"""
def best_f1_threshold(p, yt):
    pr, rc, th = precision_recall_curve(yt, p); f = 2*pr*rc/(pr+rc+1e-12)
    return float(th[np.argmax(f[:-1])])
THR = {name: best_f1_threshold(p, y_tr) for name, p in OOF.items()}
print({k: round(v, 3) for k, v in THR.items()})
""")

md("## 3.5 Final evaluation on the untouched 20% test set")
code(r"""
final = {}
FINAL_MODELS = {"Logistic Regression (2014)": models["Logistic Regression (2014)"],
                "Decision Tree (2014)": models["Decision Tree (2014)"],
                "Neural Network / MLP (2014)": models["Neural Network / MLP (2014)"],
                "Random Forest": models["Random Forest"], "LightGBM": models["LightGBM"]}
TEST_P = {}
for name, (pre, clf) in FINAL_MODELS.items():
    TEST_P[name] = Pipeline([("pre", pre), ("clf", clf)]).fit(D_tr, y_tr).predict_proba(D_te)[:, 1]
for name, mk in CANDIDATES.items():
    TEST_P[name] = mk().fit(D_tr, y_tr).predict_proba(D_te)[:, 1]
TEST_P["Final model (ours)"] = TEST_P[FINAL_NAME]

rows = []
for name, p in TEST_P.items():
    a = evaluate(p, y_te, 0.5); b = evaluate(p, y_te, THR[name])
    rows.append({"model": name, "roc_auc": a["roc_auc"], "pr_auc": a["pr_auc"], "brier": a["brier"],
                 "top10_capture": a["top10_capture"], "recall@0.5": a["recall"], "f1@0.5": a["f1"],
                 "threshold": THR[name], "precision@thr": b["precision"], "recall@thr": b["recall"],
                 "f1@thr": b["f1"], "mcc@thr": b["mcc"],
                 "precision@0.5": a["precision"], "accuracy@0.5": a["accuracy"], "bal_acc@0.5": a["balanced_accuracy"],
                 "accuracy@thr": b["accuracy"], "bal_acc@thr": b["balanced_accuracy"],
                 **{f"{k}@0.5": a[k] for k in ("tp", "fp", "fn", "tn")}, **{f"{k}@thr": b[k] for k in ("tp", "fp", "fn", "tn")}})
test = pd.DataFrame(rows).set_index("model")
display(test.round(4)); RESULTS["test_results"] = test.round(5).reset_index().to_dict("records")
RESULTS["test_set"] = {"n": int(len(y_te)), "positives": int(y_te.sum()), "always_no_accuracy": float(1 - y_te.mean())}
# Raw test-set predictions, so every reported test metric can be recomputed independently (verify_results.py)
pd.DataFrame({"y_true": y_te, **TEST_P}, index=D_te.index).to_csv(RES + "test_predictions.csv")
json.dump({"thresholds": THR}, open(RES + "thresholds.json", "w"), indent=1)
""")
code(r"""
best = "Final model (ours)"; base = "Neural Network / MLP (2014)"
# paired bootstrap on the test set: ours vs the 2014-style neural network
rng = np.random.RandomState(SEED); d_auc, d_pr = [], []
for _ in range(1000):
    i = rng.randint(0, len(y_te), len(y_te))
    if y_te[i].sum() == 0: continue
    d_auc.append(roc_auc_score(y_te[i], TEST_P[best][i]) - roc_auc_score(y_te[i], TEST_P[base][i]))
    d_pr.append(average_precision_score(y_te[i], TEST_P[best][i]) - average_precision_score(y_te[i], TEST_P[base][i]))
boot = {"n_resamples": len(d_auc), "delta_auc": float(np.mean(d_auc)), "auc_ci95": list(np.percentile(d_auc, [2.5, 97.5])),
        "p_auc": float(2 * min(np.mean(np.array(d_auc) <= 0), np.mean(np.array(d_auc) >= 0))),
        "delta_pr_auc": float(np.mean(d_pr)), "pr_auc_ci95": list(np.percentile(d_pr, [2.5, 97.5])),
        "p_pr": float(2 * min(np.mean(np.array(d_pr) <= 0), np.mean(np.array(d_pr) >= 0)))}
print(json.dumps(boot, indent=1)); RESULTS["bootstrap_ours_vs_mlp"] = boot

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for name in [base, "Logistic Regression (2014)", "Tuned XGBoost", "CatBoost", best]:
    fpr, tpr, _ = roc_curve(y_te, TEST_P[name]); ax[0].plot(fpr, tpr, lw=1.6 if name == best else 1, label=f"{name} ({roc_auc_score(y_te, TEST_P[name]):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, TEST_P[name]); ax[1].plot(rc, pr, lw=1.6 if name == best else 1, label=f"{name} ({average_precision_score(y_te, TEST_P[name]):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.7); ax[0].set(title="ROC (test set)", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=7)
ax[1].axhline(y_te.mean(), ls=":", c="k", lw=.7); ax[1].set(title="Precision–Recall (test set)", xlabel="recall", ylabel="precision"); ax[1].legend(fontsize=7)
cm = confusion_matrix(y_te, (TEST_P[best] >= THR[best]).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[2], cbar=False, xticklabels=["no", "yes"], yticklabels=["no", "yes"])
ax[2].set(title=f"Confusion matrix — ours (t = {THR[best]:.2f})", xlabel="predicted", ylabel="actual")
save("lab4_test_curves.png")

# cumulative gains / lift
fig, ax = plt.subplots(figsize=(6, 4.2))
for name in [base, best]:
    o = np.argsort(-TEST_P[name]); g = np.cumsum(y_te[o]) / y_te.sum()
    ax.plot(np.arange(1, len(o)+1) / len(o) * 100, g * 100, label=name)
ax.plot([0, 100], [0, 100], "k:", lw=.7, label="random calling")
ax.set(xlabel="% of clients called (highest score first)", ylabel="% of subscribers reached", title="Cumulative gains (test set)")
ax.legend(); save("lab4_gains.png")
gains50 = {n: float(y_te[np.argsort(-TEST_P[n])[: len(y_te)//2]].sum() / y_te.sum()) for n in [base, best]}
gains20 = {n: float(y_te[np.argsort(-TEST_P[n])[: len(y_te)//5]].sum() / y_te.sum()) for n in [base, best]}
RESULTS["capture_top20pct"] = gains20; print("top 20%:", gains20)
RESULTS["capture_top50pct"] = gains50; print("buyers reached by calling best 50%:", gains50)
""")

md("## 3.6 Leakage check — what `duration` would add (not deployable)")
code(r"""
leak = {"duration_alone_auc_test": roc_auc_score(y_te, dur_te)}
Dd_tr, Dd_te = D_tr.assign(duration=dur_tr), D_te.assign(duration=dur_te)
NUMERIC_D = NUMERIC + ["duration"]
pre_d = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC_D),
                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL)])
pd_ = Pipeline([("pre", pre_d), ("clf", lgbm(**BEST_LGB))]).fit(Dd_tr, y_tr).predict_proba(Dd_te)[:, 1]
leak["with_duration_auc_test"] = roc_auc_score(y_te, pd_)
leak["without_duration_auc_test"] = roc_auc_score(y_te, TEST_P["Tuned LightGBM"])
print({k: round(v, 4) for k, v in leak.items()}); RESULTS["leakage"] = leak
""")
md("`duration` alone ranks clients about as well as the whole published 2014 model, and adding it pushes AUC above 0.9. But the call length is only known after the call, so a model that uses it cannot choose whom to call. All results reported as **ours** exclude it.")

# ------------------------------------------------------------ LAB 5
md("# 4. Lab 5 — RFM features and K-Means customer segmentation")
md(r"""
Adapting RFM to telemarketing:
* **Recency** = days since the client was last contacted in a previous campaign (never contacted → a large value, 999)
* **Frequency** = total number of contacts (this campaign + previous campaigns)
* **Monetary** = average yearly account balance

`age` is added as profile context. Recency, frequency and monetary are log-transformed, then all four are standardised. The number of clusters is chosen with the elbow (knee) rule and checked with the silhouette score. The target `y` is **not** used for clustering; it is only used afterwards to interpret the clusters.
""")
code(r"""
rfm = pd.DataFrame({"recency": np.where(df.pdays >= 0, df.pdays, 999),
                    "frequency": df.campaign.clip(upper=CAPS["campaign"][1]) + df.previous.clip(upper=CAPS["previous"][1]),
                    "monetary": df.balance.clip(*CAPS["balance"]),
                    "age": df.age})
rfm_t = pd.DataFrame({"recency": np.log1p(rfm.recency), "frequency": np.log1p(rfm.frequency),
                      "monetary": np.sign(rfm.monetary) * np.log1p(np.abs(rfm.monetary)), "age": rfm.age})
Zr = StandardScaler().fit_transform(rfm_t)
sub = np.random.RandomState(SEED).choice(len(Zr), 8000, replace=False)
inertia, sil = [], []
for k in range(2, 11):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Zr)
    inertia.append(km.inertia_); sil.append(silhouette_score(Zr[sub], km.labels_[sub]))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(range(2, 11), inertia, marker="o"); ax[0].set(title="Elbow method", xlabel="k", ylabel="inertia (WCSS)")
ax[1].plot(range(2, 11), sil, marker="o", c="#d1603d"); ax[1].set(title="Silhouette score", xlabel="k")
save("lab5_elbow.png")
print({k: round(s, 3) for k, s in zip(range(2, 11), sil)})
# Knee of the elbow curve: the k farthest from the straight line joining the first and last points (normalised).
ks = np.arange(2, 11); xn = (ks - ks[0]) / (ks[-1] - ks[0])
yn = (np.array(inertia) - min(inertia)) / (max(inertia) - min(inertia))
K_ELBOW = int(ks[np.argmax((1 - xn) - yn)]); K_SIL = int(ks[np.argmax(sil)])
print("elbow (knee) k =", K_ELBOW, "| best-silhouette k =", K_SIL)
RESULTS["kmeans_selection"] = {"k_elbow": K_ELBOW, "k_best_silhouette": K_SIL,
                               "inertia": dict(zip(map(int, ks), map(float, inertia))), "silhouette": dict(zip(map(int, ks), map(float, sil)))}
""")
code(r"""
K = K_ELBOW   # chosen by the elbow (knee) rule; its silhouette is reported alongside
km = KMeans(K, n_init=20, random_state=SEED).fit(Zr)
seg = rfm.assign(cluster=km.labels_, subscribed=(df.y == "yes").astype(int),
                 never_contacted=(df.pdays == -1).astype(int), prior_success=(df.poutcome == "success").astype(int))
prof = seg.groupby("cluster").agg(size=("age", "size"), recency_median=("recency", "median"),
                                  frequency_mean=("frequency", "mean"), monetary_median=("monetary", "median"),
                                  age_mean=("age", "mean"), never_contacted=("never_contacted", "mean"),
                                  prior_success=("prior_success", "mean"), conversion=("subscribed", "mean"))
prof["share"] = prof["size"] / len(seg)
prof = prof.sort_values("conversion", ascending=False); display(prof.round(3))
RESULTS["kmeans"] = {"k": K, "silhouette": float(silhouette_score(Zr[sub], km.labels_[sub])), "silhouette_sample": len(sub),
                     "profiles": prof.round(4).reset_index().to_dict("records")}

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
prof["conversion"].mul(100).plot.bar(ax=ax[0], color="#2e75b6"); ax[0].axhline(11.7, ls="--", c="k", lw=.8)
ax[0].set(title="Conversion rate by cluster", ylabel="%")
Z2 = PCA(2, random_state=SEED).fit_transform(Zr)
sc = ax[1].scatter(Z2[sub, 0], Z2[sub, 1], c=km.labels_[sub], cmap="tab10", vmin=0, vmax=9, s=4, alpha=.6)
ax[1].legend(*sc.legend_elements(), title="cluster", fontsize=7, loc="lower left")
ax[1].set(title="Clusters in PCA space", xlabel="PC1", ylabel="PC2")
save("lab5_clusters.png")
""")
code(r"""
# Automatic naming + strategy from the profile (so the text always matches the numbers)
def describe(r):
    if r.never_contacted < .5 and r.prior_success > .1:
        return "Warm, previously converted", "Priority call list: call first, assign senior agents, offer renewal/upgrade rates."
    if r.never_contacted < .5:
        return "Recently re-contacted", "Second-touch campaign: reference the previous offer, use cellular channel."
    if r.frequency_mean > prof.frequency_mean.median() * 1.5:
        return "Over-called cold leads", "Cap call attempts at 3; move to email/SMS; avoid wasting agent time."
    if r.monetary_median < 100:
        return "Low/negative-balance cold prospects", "Lowest priority: cheap digital outreach only; call only when the model score is high."
    if r.age_mean >= 45:
        return "Older cold prospects with savings", "First-contact campaign with a secure, fixed-return deposit offer; prefer cellular contact and high-conversion months."
    return "Young cold prospects", "Mobile-first contact; starter deposit products; limit repeat calls to 2–3 attempts."
strat = pd.DataFrame([dict(cluster=c, **dict(zip(["segment", "strategy"], describe(r))),
                           share=r.share, conversion=r.conversion) for c, r in prof.iterrows()])
dup = strat.segment.duplicated(keep=False)   # keep names unique if two clusters fall under the same rule
strat.loc[dup, "segment"] = strat.loc[dup].apply(lambda r: f"{r.segment} (cluster {r.cluster})", axis=1)
display(strat); RESULTS["kmeans_strategy"] = strat.round(4).to_dict("records")

# Does the cluster label help the supervised model?
D_tr_c = D_tr.assign(cluster=km.labels_[D_tr.index].astype(str));
pre_c = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC),
                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL + ["cluster"])])
auc_c = cross_val_score(Pipeline([("pre", pre_c), ("clf", lgbm(**BEST_LGB))]), D_tr_c, y_tr, cv=cv5, scoring="roc_auc").mean()
auc_nc = cross_val_score(Pipeline([("pre", make_pre(False)), ("clf", lgbm(**BEST_LGB))]), D_tr, y_tr, cv=cv5, scoring="roc_auc").mean()
print(f"LightGBM CV AUC without cluster {auc_nc:.4f} | with cluster feature {auc_c:.4f}")
RESULTS["cluster_as_feature"] = {"without": auc_nc, "with": auc_c}
""")

# ------------------------------------------------------------ LAB 7
md(r"""
# 5. Lab 7 — Genetic Algorithm feature selection
The GA searches over **feature groups** (each raw or engineered attribute, with all of its one-hot columns kept together), so a chromosome is a bit-string of length = number of attributes.

* **Fitness** = 3-fold CV ROC-AUC of a fast LightGBM on the training set − 0.002 × (number of selected attributes), which rewards accuracy and penalises complexity
* **Selection** = tournament (size 3); **crossover** = uniform (p = 0.8); **mutation** = bit-flip (p = 1/L); **elitism** = best 2 kept
* Population 20, 15 generations, with fitness caching
""")
code(r"""
GROUPS = NOMINAL + NUMERIC
L = len(GROUPS); print(L, "attributes in the search space")
cv3 = StratifiedKFold(3, shuffle=True, random_state=SEED)
_cache = {}
def fitness(mask):
    key = tuple(mask)
    if key in _cache: return _cache[key]
    if mask.sum() == 0: return (0.0, 0.0)
    cols = [g for g, m in zip(GROUPS, mask) if m]
    nom, num = [c for c in cols if c in NOMINAL], [c for c in cols if c in NUMERIC]
    tr = []
    if num: tr.append(("num", SimpleImputer(strategy="median"), num))
    if nom: tr.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nom))
    pipe = Pipeline([("pre", ColumnTransformer(tr)), ("clf", lgbm(n_estimators=150, learning_rate=.08))])
    auc = cross_val_score(pipe, D_tr[cols], y_tr, cv=cv3, scoring="roc_auc").mean()
    _cache[key] = (auc - 0.002 * mask.sum(), auc); return _cache[key]

rng = np.random.RandomState(SEED)
POP, GENS, ELITE = 20, 15, 2
pop = (rng.rand(POP, L) < .6).astype(int); pop[0] = 1   # include the all-features individual
history = []; t0 = time.time()
for g in range(GENS):
    fit = np.array([fitness(ind)[0] for ind in pop])
    order = np.argsort(-fit); history.append((g, fit.max(), fit.mean(), int(pop[order[0]].sum())))
    new = [pop[i].copy() for i in order[:ELITE]]
    while len(new) < POP:
        p1 = pop[max(rng.choice(POP, 3), key=lambda i: fit[i])]
        p2 = pop[max(rng.choice(POP, 3), key=lambda i: fit[i])]
        child = np.where(rng.rand(L) < .5, p1, p2) if rng.rand() < .8 else p1.copy()
        flip = rng.rand(L) < 1 / L; child[flip] = 1 - child[flip]
        new.append(child)
    pop = np.array(new)
    print(f"gen {g:2d}  best fitness {fit.max():.4f}  mean {fit.mean():.4f}  n_attr {history[-1][3]}")
fit = np.array([fitness(ind)[0] for ind in pop]); BEST_MASK = pop[np.argmax(fit)]
GA_COLS = [g for g, m in zip(GROUPS, BEST_MASK) if m]
ga_time = time.time() - t0
print(f"\nGA finished in {ga_time:.0f}s, {len(_cache)} unique subsets evaluated")
print(f"Selected {len(GA_COLS)}/{L} attributes:", GA_COLS)
print("Dropped:", [g for g in GROUPS if g not in GA_COLS])

h = pd.DataFrame(history, columns=["gen", "best", "mean", "n_attr"])
fig, ax = plt.subplots(figsize=(6, 3.6)); ax.plot(h.gen, h.best, marker="o", label="best"); ax.plot(h.gen, h["mean"], marker=".", label="population mean")
ax.set(xlabel="generation", ylabel="fitness (AUC − penalty)", title="GA convergence"); ax.legend(); save("lab7_ga_convergence.png")
RESULTS["ga"] = {"selected": GA_COLS, "dropped": [g for g in GROUPS if g not in GA_COLS], "n_selected": len(GA_COLS),
                 "n_total": L, "seconds": ga_time, "evaluations": len(_cache), "history": h.round(5).to_dict("records")}
""")
code(r"""
# Compare subsets on the held-out test set with the tuned LightGBM
def subset_model(cols):
    nom, num = [c for c in cols if c in NOMINAL], [c for c in cols if c in NUMERIC]
    tr = ([("num", SimpleImputer(strategy="median"), num)] if num else []) + \
         ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nom)] if nom else [])
    return Pipeline([("pre", ColumnTransformer(tr)), ("clf", lgbm(**BEST_LGB))])

l1 = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(penalty="l1", C=.01, solver="saga", max_iter=3000, tol=1e-3))]).fit(D_tr, y_tr)
pre_tr = make_pre().fit(D_tr)
grp_tr = NUMERIC + [g for g, cats in zip(NOMINAL, pre_tr.named_transformers_["cat"].categories_) for _ in cats]
kept_groups = set(np.array(grp_tr)[np.abs(l1["clf"].coef_[0]) > 1e-6])
L1_COLS = [g for g in GROUPS if g in kept_groups]
grp_mi = pd.Series(mutual_info_classif(pre_tr.transform(D_tr), y_tr, random_state=SEED)).groupby(np.array(grp_tr)).sum()
MI_COLS = list(grp_mi.sort_values(ascending=False).index[: len(GA_COLS)])

sel_rows = []
for name, cols in [("All attributes", GROUPS), ("L1-selected", L1_COLS), (f"Mutual-info top-{len(GA_COLS)}", MI_COLS), ("GA-selected (ours)", GA_COLS)]:
    t = time.time(); m = subset_model(cols).fit(D_tr[cols], y_tr); fit_s = time.time() - t
    p = m.predict_proba(D_te[cols])[:, 1]
    sel_rows.append({"subset": name, "n_attributes": len(cols), "n_encoded_cols": m["pre"].transform(D_tr[cols].head(5)).shape[1],
                     "test_roc_auc": roc_auc_score(y_te, p), "test_pr_auc": average_precision_score(y_te, p), "fit_seconds": fit_s})
sel = pd.DataFrame(sel_rows).set_index("subset"); display(sel.round(4))
print("L1:", L1_COLS); print("MI:", MI_COLS)
RESULTS["l1_subset"] = L1_COLS; RESULTS["mi_subset"] = MI_COLS
RESULTS["feature_selection_test"] = sel.round(5).reset_index().to_dict("records")
""")

# ------------------------------------------------------------ SHAP
md("# 6. Explainability — TreeSHAP values of the tuned LightGBM\nLightGBM computes exact TreeSHAP values natively (`pred_contrib=True`). One-hot columns are summed back to their original attribute.")
code(r"""
shap_model = Pipeline([("pre", make_pre(False)), ("clf", lgbm(**BEST_LGB))]).fit(D_tr, y_tr)
Xte_enc = shap_model["pre"].transform(D_te)
contrib = shap_model["clf"].booster_.predict(Xte_enc, pred_contrib=True)[:, :-1]
# Parent attribute by position: numeric columns map to themselves, one-hot columns to their nominal attribute.
shap_cats = shap_model["pre"].named_transformers_["cat"].categories_
shap_grp = NUMERIC + [g for g, cats in zip(NOMINAL, shap_cats) for _ in cats]
assert len(shap_grp) == contrib.shape[1]
S = pd.DataFrame(contrib).T.groupby(np.array(shap_grp)).sum().T
imp = S.abs().mean().sort_values(ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
imp.head(15).iloc[::-1].plot.barh(ax=ax[0], color="#2e75b6"); ax[0].set(title="Mean |SHAP| (log-odds), top 15 attributes")
for i, f in enumerate(imp.head(10).index[::-1]):
    v = S[f].values; jitter = np.random.RandomState(SEED + i).uniform(-.25, .25, len(v))
    if f in NUMERIC:   # colour by the feature's value (percentile rank); fixed 0–1 scale
        ax[1].scatter(v, i + jitter, c=D_te[f].rank(pct=True).fillna(.5).values, cmap="coolwarm", vmin=0, vmax=1, s=2, alpha=.4)
    else:              # nominal attribute: no high/low ordering, so plot in grey
        ax[1].scatter(v, i + jitter, color="#8a8f98", s=2, alpha=.4)
ax[1].set_yticks(range(10)); ax[1].set_yticklabels(imp.head(10).index[::-1]); ax[1].axvline(0, c="k", lw=.6)
ax[1].set(title="SHAP value per client (red = high, blue = low value; grey = categorical)", xlabel="SHAP value")
save("shap_summary.png")
RESULTS["shap_top15"] = imp.head(15).round(5).to_dict()
""")

md("# 7. Summary")
code(r"""
s = test.loc["Final model (ours)"]; b = test.loc["Neural Network / MLP (2014)"]
summary = pd.DataFrame({
    "2014-style neural network": [b.roc_auc, b.pr_auc, b["recall@0.5"], b["f1@0.5"], b.top10_capture, gains50[base]],
    "Our final model (" + FINAL_NAME + ")": [s.roc_auc, s.pr_auc, s["recall@thr"], s["f1@thr"], s.top10_capture, gains50[best]]},
    index=["ROC-AUC", "PR-AUC", "Recall (NN at 0.5, ours at tuned threshold)", "F1 (NN at 0.5, ours at tuned threshold)", "Buyers in top 10% of calls", "Buyers in top 50% of calls"])
display(summary.round(4))
RESULTS["headline"] = summary.round(5).to_dict()
json.dump(RESULTS, open(RES + "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
test.round(5).to_csv(RES + "test_results.csv"); cmp.round(5).to_csv(RES + "cv_model_comparison.csv")
print("saved results/results.json")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Bank_Marketing_ML_Labs.ipynb"); print(len(cells), "cells")
