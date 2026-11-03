# Where each objective is implemented

Project: Handwritten Modi Character Recognition using Handcrafted Features (Group TY-A2)

All paths are inside the project folder `modi-lipi-xai`. To train and test everything for objectives 2, 3 and 4, run `python scripts\run_lookalike.py`.

## Objective 1: Recognise 57 Modi characters with one model, an SVM on handcrafted features

Base papers: Mandke et al. 2024, Anam & Gupta 2015, Kulkarni et al. 2014

| What | Where |
|---|---|
| Cleaning with Otsu thresholding | `src/preprocess.py`, `binarize()` (line 121) |
| HOG features (stroke direction, 1,764 values) | `src/features/hog_features.py`, `extract()` (line 48) |
| Zernike moments (overall shape, 49 values) | `src/features/zernike_features.py`, `extract()` (line 43) |
| LBP features (small local patterns, 448 values) | `src/features/lbp_features.py`, `extract()` (line 76) |
| Joining the three into 2,261 values | `src/features/fusion.py`, `extract_one()` (line 89) |
| Training the SVM on those features, after LDA | `src/lookalike.py`, `fit_svm()` (line 89) |
| Testing what each feature adds | `scripts/run_training.py`, `run_ablation()` (line 191) |
| Result files | `results/metrics/lookalike.json`, `results/metrics/ablation.json` |
| In the app | Results page, Model tab |

What we got: the SVM alone reads 86.36% of the 11,400 test images correctly, against the 77% Mandke et al. report for their SVM (a different dataset, so it is only a rough comparison). Without HOG it falls to 81.4%, without LBP to 83.9%, without Zernike only to 86.0%.

## Objective 2: Find the pairs the SVM mixes up most, and use SHAP to see what separates them

Base papers: Kolte & Gupta 2025, Anam & Gupta 2015, Ekbote et al. 2022

| What | Where |
|---|---|
| The SVM's guesses on held-out training images | `scripts/run_lookalike.py`, `oof_scores()` (line 34) |
| Counting errors to get the 15 most confused pairs | `scripts/run_lookalike.py`, `main()` (line 84) |
| SHAP ranking of the 35 regions for one pair | `src/lookalike.py`, `rank_groups()` (line 100) |
| Picking the top 10 regions | `src/lookalike.py`, `choose_columns()` (line 129) |
| Most confused pairs on the test set | `src/evaluate.py`, `most_confused_pairs()` (line 58) |
| SHAP on the main SVM, and for confused pairs | `src/explain.py`, `run_global()` (line 253), `src/explain.py`, `contrast_pairs()` (line 367) |
| Result files | `results/metrics/confused_pairs_final.json`, `results/shap/confused_pairs_contrast.json` |
| In the app | Explain page, Look-alikes tab; Results page, Errors tab |

What we got: 15 candidate pairs from training data. For i and ii, SHAP points to the top right of the character, which is where the extra hook of ii sits.

## Objective 3: Build a small specialist for each pair, and let it decide when the SVM is stuck between them

This is the new idea in our project. Base paper: Kolte & Gupta 2025.

| What | Where |
|---|---|
| Training one specialist on its 10 regions | `src/lookalike.py`, `fit_specialist()` (line 135) |
| Checking it on held-out training images | `scripts/run_lookalike.py`, `test_pair_oof()` (line 45) |
| Keeping it only if it beats the SVM there | `scripts/run_lookalike.py`, `main()` (line 92) |
| Top two guesses of the SVM | `src/lookalike.py`, `top_two()` (line 69) |
| The rule: if the top two are a kept pair, the specialist decides | `src/lookalike.py`, `decide()` (line 73) |
| Using it to read a character | `src/predict.py`, `recognize()` (line 130) |
| Saved model | `models/main_svm_lookalike.joblib` |
| In the app | Read page: tap a character, and it says when a specialist decided |

What we got: 12 of the 15 specialists passed the training check and were kept.

## Objective 4: Check on a fixed test set whether the specialists really reduce mistakes

Base paper: Bhong & Kulkarni 2025

| What | Where |
|---|---|
| Scoring the test set | `scripts/run_lookalike.py`, `main()` (line 104) |
| McNemar test on paired predictions | `scripts/run_lookalike.py`, `mcnemar()` (line 61) |
| Result files | `results/metrics/lookalike.json`, `results/metrics/per_class_final.json` |
| In the app | Results page, Look-alikes tab |

What we got: 86.36% with the SVM alone and 86.78% with the specialists. They fixed 182 images and broke 134 (McNemar p = 0.008). Pair by pair, on the images each specialist decided:

| Pair | Images | SVM right | Final right | Change |
|---|---|---|---|---|
| i / ii | 374 | 272 | 291 | +19 |
| five / six | 298 | 247 | 261 | +14 |
| k / m | 245 | 203 | 212 | +9 |
| dha / tha | 147 | 128 | 135 | +7 |
| th / thh | 217 | 174 | 180 | +6 |
| b / chh | 105 | 87 | 93 | +6 |
| d / th | 196 | 157 | 162 | +5 |
| o / ou | 335 | 309 | 313 | +4 |
| ai / e | 241 | 225 | 227 | +2 |
| r / th | 217 | 181 | 180 | -1 |
| h / thha | 251 | 158 | 153 | -5 |
| ja / n | 310 | 261 | 253 | -8 |

Nine pairs improve and three get worse.
