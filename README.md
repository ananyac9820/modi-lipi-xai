# Modi Lipi XAI

**Explainable recognition of handwritten Modi script using classical machine learning, with no deep learning.**

> 🚧 Work in progress. This README is a stub. Results, the architecture diagram and the full
> reference list get filled in at milestone 10.

## What is Modi script?

Modi (मोडी) is a cursive script that was used to write Marathi for about 700 years. It appears in
land records, court documents and administrative letters across Maharashtra. Very few people can
read it today, so millions of archived documents are effectively locked away.

## Why classical ML?

Ten papers surveyed (2021–2024) fall into two camps:

- **Deep learning** (CNN, VGG16, ResNet, InceptionV3, LSTM) reports 86–98.92% accuracy, with clear
  signs of overfitting. A shallow ResNet9 (98.92%) beats ResNet50 (91.91%) and InceptionV3 (86%),
  which is an inverted ranking on a benchmark of only ~90 images per class. Another paper reports
  97.6% train vs 87.3% validation accuracy.
- **Classical** (Zernike + DCT with KNN; HOG alone; wavelet + SVD with Euclidean distance) reports
  97.77–99.3%, but all three use **private datasets of undisclosed size**, and one covers only 27 of
  46 characters.

### Gaps this project targets

1. **Three-family descriptor fusion.** HOG (stroke direction) + Zernike moments (rotation-invariant
   global shape) + LBP (local texture at stroke junctions). None of the surveyed papers uses LBP.
2. **A stacking ensemble** (Random Forest + SVM-RBF + XGBoost) in place of a single weak classifier.
3. **Explainability** with SHAP. None of the surveyed papers explains its predictions.
4. **Per-class error analysis**: a full confusion matrix and a table of the most-confused pairs.

## Setup (Python 3.11+, tested on 3.14, CPU only)

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Pipeline

| Stage | Script | Status |
|---|---|---|
| Download data | `python scripts/download_data.py` | milestone 2 |
| Preprocess | `python scripts/run_preprocessing.py` | milestone 2 |
| Feature extraction | `python scripts/run_feature_extraction.py` | milestone 3 |
| Training + ablation | `python scripts/run_training.py` | milestone 5 |
| Evaluation + SHAP | `python scripts/run_evaluation.py` | milestones 5–6 |
| Web app | `streamlit run app/app.py` | milestones 8–9 |

All tunable parameters live in [`src/config.py`](src/config.py). The random seed is fixed at 42, so
results are reproducible.

## Repository layout

```
docs/       literature survey, synopsis, reference papers, final report
data/       raw / processed images and cached features (gitignored)
src/        pipeline code (preprocess, segmentation, features, reduce, models, evaluate, explain, predict)
scripts/    standalone entry points, one per stage
notebooks/  exploration, feature analysis, results analysis
models/     trained models (gitignored)
results/    figures, metrics JSON, SHAP outputs
app/        Streamlit web app
```

## References

The ten surveyed papers will be listed here (see `docs/03_reference_papers/`).
