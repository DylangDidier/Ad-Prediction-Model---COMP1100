# Predicting Ad Clicks with a Neural Network

Team repo for our AI course final project: Vaughn Miller and Dylan Didier, Fall 2026.

We train a neural network on real mobile ad impressions (Avazu, Kaggle) to predict whether someone clicks an ad. Then we check whether it ranks ads better than the simple historical-average click rate most advertisers use.

## Setup

Clone this repo (PyCharm: **Get from VCS**, paste the URL).

In a terminal inside the repo folder (you don't have to use venv, whatever env manager you like works just fine):

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch, pandas, sklearn, matplotlib; print('ok')"
```

PyCharm: **Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Virtualenv Environment > Existing**, then pick `.venv`.

Right-click the `src` folder > **Mark Directory as > Sources Root**.

Get the data: sign in to Kaggle, accept the rules for the [Avazu CTR Prediction competition](https://www.kaggle.com/c/avazu-ctr-prediction), download `train.gz` (about 1.2 GB, about 40M rows), and put it in `data/raw/`. Leave it zipped, the script reads the `.gz` directly.

Run it: right-click `src/ctr_mid_report.py` > **Run**. Reading the file takes several minutes, and training takes a few more on a laptop CPU. Everything it makes lands in `outputs/`.

If you need a new package: add it to `requirements.txt`, commit, tell your teammate, then run `pip install -r requirements.txt`.

## Folder structure

| Folder | What goes there | In git? |
| --- | --- | --- |
| `data/raw/` | `train.gz` from Kaggle | no (too big; everyone downloads their own) |
| `src/` | shared scripts: sampling, features, baseline, model, evaluation | yes |
| `notebooks/<your name>/` | your own scratch notebooks | yes, but nobody else runs them |
| `outputs/` | `results.csv`, `loss_curve.png`, `click_capture.png`, `feature_importance.*`, and the `ctr_by_*.csv` tables for Excel | yes |
| `excel/` | the Excel workbook with the baseline pivot tables and charts | yes |
| `report/` | the proposal, mid-project report, and final report | yes |
| `poster/` | the one-slide poster | yes |

## What the script does

| Step | What happens |
| --- | --- |
| 1. Sample | Reads `train.gz` in 1M-row chunks and keeps 1.25% of each (about 500k rows). The last day (2014-10-30) is the test set, and 10% of the rest is kept aside for validation. |
| 2. Features | Splits `hour` into hour of day and day of week, maps each category to an ID (values seen fewer than 10 times become "other"), and drops `device_id` and `device_ip`. |
| 3. Baseline | Predicts each test ad's click rate from the past average for its site category, device type, and banner position, which is the Excel pivot-table approach. |
| 4. Model | Gives each of the 21 features an 8-number embedding, then two dense layers (128 and 64) and a sigmoid output. Trained with Adam and stops early once validation loss stops improving. |
| 5. Evaluate | Compares AUC, log loss, and the share of clicks caught in the top 10% of predictions on the test day. |
| 6. Failure check | Compares AUC on sites that were common in training with sites that were rare or new. |

## Feature importance

After `ctr_mid_report.py` has run once (so the sample is cached), right-click `src/feature_importance.py` > **Run**. It rebuilds the same features, trains the same model (or reuses the one saved in `data/cache/`), then shuffles one feature at a time on the test day and measures how much AUC drops and log loss rises. It repeats each shuffle 5 times. Features that carry the same information (like `site_id` and `site_domain`) can cover for each other when shuffled alone, so it also shuffles related features together as groups (site, app, device, time, the anonymous C columns).

Output: `outputs/feature_importance.csv` (per feature), `outputs/feature_importance_groups.csv` (per group), and `outputs/feature_importance.png` (both as bar charts, for the final report and poster).

`day_of_week` always scores 0: the test set is a single day, so every test row has the same day of week and shuffling it changes nothing. Its effect can't be measured with this test split.

Knobs (`SAMPLE_FRAC`, `EMB_DIM`, `HIDDEN`, `DROPOUT`, `LR`, and so on) are at the top of the script. Set `SAMPLE_FRAC = 0.005` for the smaller 200k-row sample used in the mid-report.

Don't report accuracy. Most ads aren't clicked, so "never clicks" looks accurate and is useless.

## Mid-project report checklist

- [ ] `train.gz` in `data/raw/` on both machines
- [ ] Script runs end to end, and `outputs/` has `results.csv` and both charts
- [ ] Excel charts made from the `ctr_by_*.csv` tables
- [ ] Model diagram (features > embeddings > dense layers > click probability)
- [ ] Screenshot of the script running
- [ ] What works, what doesn't, and what's left written down
- [ ] Report is 2–3 pages, saved as `LastName_LastName_MidReport.pdf`

## Final report checklist

- [ ] Edit this

## Submission checklist

- results in the report match `outputs/results.csv` from the latest run
- the test day is the held-out last day, never used for training or tuning
- every chart has labeled axes and units
- data source, libraries, and AI tools are cited, and the AI Use Statement is included
- file names follow `LastName_LastName_<Part>.pdf` and `LastName_LastName_ProjectFiles.zip`

