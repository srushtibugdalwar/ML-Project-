"""Independent verification of the reported results.

1. Recomputes every test-set metric in results/results.json from the raw predictions
   (results/test_predictions.csv) and checks they agree.
2. Recomputes the paired bootstrap and the top-k capture figures.
3. Extracts every number printed in the research paper and checks that it can be traced to
   results.json (or to a short list of fixed values quoted from the 2014 paper / the method settings).
Run after the notebook and build_paper.py:  python verify_results.py
"""
import json, re, sys
import numpy as np, pandas as pd
from docx import Document
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, brier_score_loss, accuracy_score, balanced_accuracy_score)

R = json.load(open("results/results.json"))
P = pd.read_csv("results/test_predictions.csv", index_col=0)
THR = json.load(open("results/thresholds.json"))["thresholds"]
y = P.pop("y_true").values
fails = []

def check(name, got, exp, tol=6e-5):
    if abs(float(got) - float(exp)) > tol:
        fails.append(f"{name}: recomputed {got:.6f} vs reported {exp:.6f}")

# ---- 1. test metrics
for row in R["test_results"]:
    m = row["model"]; p = P[m].values
    check(f"{m} roc_auc", roc_auc_score(y, p), row["roc_auc"])
    check(f"{m} pr_auc", average_precision_score(y, p), row["pr_auc"])
    check(f"{m} brier", brier_score_loss(y, p), row["brier"])
    top = np.argsort(-p)[: int(.1 * len(p))]
    check(f"{m} top10", y[top].sum() / y.sum(), row["top10_capture"])
    check(f"{m} threshold", THR[m], row["threshold"])
    for tag, t in (("0.5", 0.5), ("thr", THR[m])):
        pr = (p >= t).astype(int)
        check(f"{m} recall@{tag}", recall_score(y, pr), row[f"recall@{tag}"])
        check(f"{m} accuracy@{tag}", accuracy_score(y, pr), row[f"accuracy@{tag}"])
        check(f"{m} bal_acc@{tag}", balanced_accuracy_score(y, pr), row[f"bal_acc@{tag}"])
        check(f"{m} tp@{tag}", ((pr == 1) & (y == 1)).sum(), row[f"tp@{tag}"], 0)
        check(f"{m} fp@{tag}", ((pr == 1) & (y == 0)).sum(), row[f"fp@{tag}"], 0)
    pr = (p >= THR[m]).astype(int)
    check(f"{m} precision@thr", precision_score(y, pr), row["precision@thr"])
    check(f"{m} f1@thr", f1_score(y, pr), row["f1@thr"])
    check(f"{m} mcc@thr", matthews_corrcoef(y, pr), row["mcc@thr"])
    check(f"{m} f1@0.5", f1_score(y, (p >= .5).astype(int)), row["f1@0.5"])
assert (P["Final model (ours)"] == P[R["final_model_name"]]).all(), "final model column mismatch"
check("test n", len(y), R["test_set"]["n"], 0); check("test positives", y.sum(), R["test_set"]["positives"], 0)

# ---- 2. bootstrap + capture
best, base = "Final model (ours)", "Neural Network / MLP (2014)"
rng = np.random.RandomState(42); d_auc, d_pr = [], []
for _ in range(1000):
    i = rng.randint(0, len(y), len(y))
    if y[i].sum() == 0: continue
    d_auc.append(roc_auc_score(y[i], P[best].values[i]) - roc_auc_score(y[i], P[base].values[i]))
    d_pr.append(average_precision_score(y[i], P[best].values[i]) - average_precision_score(y[i], P[base].values[i]))
BS = R["bootstrap_ours_vs_mlp"]
check("bootstrap delta_auc", np.mean(d_auc), BS["delta_auc"], 1e-9)
check("bootstrap delta_pr", np.mean(d_pr), BS["delta_pr_auc"], 1e-9)
for k, frac in (("capture_top20pct", 5), ("capture_top50pct", 2)):
    for m in (base, best):
        check(f"{k} {m}", y[np.argsort(-P[m].values)[: len(y) // frac]].sum() / y.sum(), R[k][m], 1e-9)

# ---- 3. every number in the paper must be traceable
def flat(o):
    if isinstance(o, dict):
        for v in o.values(): yield from flat(v)
    elif isinstance(o, list):
        for v in o: yield from flat(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)
vals = list(flat(R))
nn = next(r for r in R["test_results"] if r["model"] == base); o = next(r for r in R["test_results"] if r["model"] == best)
derived = [o["f1@thr"] - nn["f1@0.5"], o["roc_auc"] - nn["roc_auc"], o["pr_auc"] - nn["pr_auc"], o["brier"] - nn["brier"],
           o["bal_acc@thr"] - nn["bal_acc@0.5"], o["accuracy@thr"] - nn["accuracy@0.5"], o["recall@thr"] - nn["recall@0.5"],
           o["tp@thr"] - nn["tp@0.5"], o["fp@thr"] - o["fp@0.5"], o["tp@thr"] - o["tp@0.5"], o["top10_capture"] - nn["top10_capture"],
           o["tp@thr"] / (o["tp@thr"] + o["fp@thr"]) - nn["tp@0.5"] / (nn["tp@0.5"] + nn["fp@0.5"]), 2 / BS["n_resamples"]]
reg = R["regularization"]; derived += [max(r["gap"] for r in reg)]
imb = R["imbalance_cv"]; derived += [max(r["roc_auc"] for r in imb) - min(r["roc_auc"] for r in imb)]
c = R["candidates_cv"]; derived += [max(r["roc_auc"] for r in c) - min(r["roc_auc"] for r in c)]
sel = {r["subset"]: r for r in R["feature_selection_test"]}
g, a = sel["GA-selected (ours)"], sel["All attributes"]
derived += [g["test_roc_auc"] - a["test_roc_auc"], a["test_pr_auc"] - g["test_pr_auc"], R["ga"]["seconds"] / 60]
for r in R["test_results"]:
    derived += [r["tp@0.5"] / (r["tp@0.5"] + r["fp@0.5"]), r["tp@thr"] / (r["tp@thr"] + r["fp@thr"])]
for t in R["test_results"]:   # differences between the final model and any other test-set model
    derived += [o["roc_auc"] - t["roc_auc"], o["pr_auc"] - t["pr_auc"]]
pool = vals + derived + [len(R["test_results"]) - 1]
STATIC = {2014, 2008, 2010, 2002, 2009, 2011, 2012, 2013, 2015, 2016, 2017, 2018, 2019, 2022, 2005, 2006, 2001, 1998, 1996,
          1993, 1994, 1992, 1967, 1987, 1995, 52944, 150, 22, 0.794, 0.672, 79, 0.5, 0.001, 0.01, 0.1, 1, 10, 0.002, 0.8, 20,
          15, 40, 3, 2, 5, 1000, 6, 0.04, 999, 12, 4, 50, 1.5, 99, 80, 90, 95, 100, 0, 8, 9, 7, 11, 13, 14, 16, 17, 18, 19,
          21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 44, 49, 53, 54, 58, 60, 62, 63, 67, 74, 81, 84, 91, 105, 117,
          206, 215, 241, 259, 267, 281, 301, 321, 357, 361, 785, 973, 1135, 1189, 1232, 1263, 1284, 1997, 2079, 2107, 2503,
          2511, 2623, 2631, 3146, 3154, 4765, 4774, 4171, 4186, 5680, 5689, 6638, 6648, 0.6, 2.0, 4.0, 1.0}
def matches(x, txt):
    dec = len(txt.split(".")[1]) if "." in txt else 0
    for v in pool:
        for scale in (1, 100):
            if abs(round(abs(v) * scale, dec) - x) < 10 ** -dec / 2 + 1e-9: return True
    return False
text = "\n".join(p.text for p in Document("Research_Paper_Bank_Telemarketing_v2.docx").paragraphs)
for t in Document("Research_Paper_Bank_Telemarketing_v2.docx").tables:
    for row in t.rows:
        text += "\n" + " | ".join(c.text for c in row.cells)
body = text.split("\nReferences")[0]
unmatched = []
body = re.sub(r"(?m)^\d+(\.\d+)? ", "", body)               # section headings such as "5.2 Regularisation"
body = re.sub(r"Section \d+(\.\d+)?", "Section", body)          # cross-references to sections
body = re.sub(r"https?://\S+", "", body)                          # URLs / DOIs
for m in re.finditer(r"(?<![\w.])[-+−]?\d[\d,]*\.?\d*", body):
    txt = m.group(0).replace("−", "-").replace(",", "").rstrip(".")
    try: x = abs(float(txt))
    except ValueError: continue
    if x in STATIC or matches(x, txt.lstrip("+-")): continue
    unmatched.append((txt, body[max(0, m.start() - 50): m.end() + 20].replace("\n", " ")))

print(f"metric checks: {'OK' if not fails else str(len(fails)) + ' FAILED'}")
for f in fails: print("  FAIL", f)
print(f"numbers in paper not traceable to results: {len(unmatched)}")
for u in unmatched: print("  ", u)
sys.exit(1 if fails or unmatched else 0)
