# Metro Card Ridership Forecast

北方工业大学《综合交通大数据技术实践》课程项目的公开作品集版本。项目以智能卡与 NCMC 合计人次为目标，完成日需求预测、高需求分类和日型聚类。

> 公开仓库只保留本人整理的源码、测试和脱敏结果摘要，不包含课程行政材料、其他成员报告、原始数据或模型产物。

## Results

数据快照含 460 条公开日记录，观察期为 2024-10-26 至 2026-08-16，其中有 200 个自然日缺失。

| Task | Method | Independent test result |
|---|---|---|
| Daily demand forecast | Same-weekday mean over the previous four weeks | MAE 29,049.13; RMSE 52,301.85; MAPE 7.9564%; R² 0.8944 |
| High-demand classification | Logistic regression | F1 0.9149; accuracy 89.33%; ROC-AUC 0.9650 |
| Descriptive day clustering | KMeans, K=3 | Development silhouette score 0.6054 |

The largest test error occurred on 2026-05-01. The weekday-only slice had R² = -0.0812, so the project keeps failure cases and scope limits visible rather than presenting only aggregate scores.

## Repository structure

```text
src/                 data pipeline, experiment, prediction and report builders
tests/               pipeline verification
results/             compact,脱敏 result summary and predictions
simple_predict.py     small prediction entry point
run.ps1               reproducible Windows runner
```

## Run

```powershell
python -m pip install -r requirements.txt
python src/download_data.py
python -m unittest discover -s tests -q
python src/predict.py --date 2026-08-18
```

The source downloader expects the original public-data location to remain available. Historical forecasts in this repository are archived experiment outputs, not live forecasts.

## What I learned

- Separate chronological train/development/test splits to reduce leakage.
- Evaluate regression, warning classification and clustering with task-appropriate metrics.
- Keep missing-day assumptions and weak slices visible.
- Package experiment code, tests and result evidence so another person can reproduce the reasoning.

