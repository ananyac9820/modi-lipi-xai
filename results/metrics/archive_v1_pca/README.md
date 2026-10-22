# Archived run v1: PCA, before the grid-line fix

Results of the first full run (preprocessing without interior grid-line removal,
REDUCE.method = "pca"). Kept for comparison. The current results in results/metrics/
come from the re-run with grid-line removal and the reducer chosen by training-set CV.

Headline (500 training images / class, fixed 11,400-image test set):
SVM 84.50 %, stacking 81.33 %, XGBoost 78.22 %, Random Forest 74.57 %.
Learning curve (stack / SVM): 72: 69.8 / 75.3, 90: 71.6 / 76.5, 250: 77.6 / 82.0,
500: 81.3 / 84.5, 1000: 83.9 / 86.7.
