# Bank Telemarketing ML — Labs + Research Paper

Revisits Moro, Cortez & Rita (2014) on the UCI Bank Marketing data (`bank-full.csv`) with a leakage-free,
imbalance-aware pipeline: XGBoost / LightGBM / CatBoost (Optuna-tuned) and ensembles, GA feature selection,
RFM + K-Means segmentation and TreeSHAP.

## Files
| File | Purpose |
|---|---|
| `Bank_Marketing_ML_Labs.ipynb` | Main code: Labs 1, 3, 4, 5, 7 + model comparison + SHAP (executed, outputs inside) |
| `bank_marketing_ml.py` | The same code as a plain Python script |
| `make_notebook.py` | Regenerates the notebook from source |
| `build_paper.py` | Builds `Research_Paper_Bank_Telemarketing_v2.docx`; every number is read from `results/` |
| `verify_results.py` | Recomputes all test metrics from the saved predictions and checks every number in the paper |
| `results/` | `results.json`, `test_predictions.csv`, `thresholds.json`, CSV tables |
| `figures/` | All figures (PNG) |
| `requirements.txt` | Python dependencies |

## Reproduce
Put `bank-full.csv` (from https://doi.org/10.24432/C5K306) next to the notebook, then:

    pip install -r requirements.txt
    jupyter nbconvert --to notebook --execute --inplace Bank_Marketing_ML_Labs.ipynb   # ~35 min on 2 cores
    python build_paper.py
    python verify_results.py

The notebook creates `figures/` and `results/` itself.
