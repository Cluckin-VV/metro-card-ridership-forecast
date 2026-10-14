"""Reproducible traffic experiment with chronological model selection and holdout audit."""
import json
import logging
import math
import os
import platform
import sqlite3
import time
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, silhouette_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import ROOT, load_clean, make_features, cluster_features

SEED = 42
RESULTS = ROOT / 'results'


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def regressors():
    specs = {'seasonal_mean': None, 'last_available': None}
    for alpha in [10, 100, 1000]:
        specs[f'ridge_{alpha}'] = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(alpha=alpha))
    for leaf in [3, 10]:
        specs[f'rf_{leaf}'] = make_pipeline(SimpleImputer(add_indicator=True), RandomForestRegressor(
            n_estimators=220, min_samples_leaf=leaf, max_features=.8, random_state=SEED, n_jobs=2))
    specs['extra_trees'] = make_pipeline(SimpleImputer(add_indicator=True), ExtraTreesRegressor(
        n_estimators=220, min_samples_leaf=5, random_state=SEED, n_jobs=2))
    for leaves in [7, 15]:
        specs[f'hgb_{leaves}'] = make_pipeline(SimpleImputer(add_indicator=True), HistGradientBoostingRegressor(
            max_iter=160, max_leaf_nodes=leaves, min_samples_leaf=15,
            l2_regularization=10, learning_rate=.05, early_stopping=False, random_state=SEED))
    return specs


def reg_predict(name, model, x):
    if name == 'seasonal_mean':
        values = x.seasonal_baseline.to_numpy()
    elif name == 'last_available':
        values = x.last_known.to_numpy()
    else:
        # Learn a residual around a causal weekday baseline, reducing level drift.
        values = model.predict(x) + x.seasonal_baseline.to_numpy()
    return np.maximum(0, values)


def fit_reg(name, spec, x, y):
    if spec is None:
        return None
    return clone(spec).fit(x, y - x.seasonal_baseline)


def reg_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'n': len(y), 'MAE': float(mean_absolute_error(y, pred)),
            'MSE': float(mean_squared_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAPE_percent': float(np.mean(np.abs(y-pred)/y)*100),
            'WAPE_percent': float(np.sum(np.abs(y-pred))/np.sum(y)*100),
            'R2': float(r2_score(y, pred)) if len(y)>1 else None}


def cls_metrics(y, pred, prob):
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'F1': float(f1_score(y, pred, zero_division=0)),
            'macro_F1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'ROC_AUC': float(roc_auc_score(y, prob)) if len(np.unique(y))>1 else None,
            'confusion_matrix': confusion_matrix(y, pred, labels=[0,1]).tolist()}


def main():
    started = time.perf_counter()
    for d in ['results', 'models', 'data/processed', 'logs']:
        (ROOT/d).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(ROOT/'logs/experiment.log', encoding='utf-8'), logging.StreamHandler()])
    data, audit = load_clean()
    save_json(RESULTS/'data_audit.json', audit)
    data.to_csv(ROOT/'data/processed/clean_daily.csv', encoding='utf-8-sig')
    data.loc[data.qr_gap != 0].to_csv(RESULTS/'qr_accounting_cases.csv', encoding='utf-8-sig')
    x_all, y_all = make_features(data)
    # Four-week warmup is a predeclared minimum history, not filtered by outcome.
    usable = y_all.notna() & x_all.seasonal_baseline.notna() & ((x_all.index-data.index.min()).days >= 28)
    x, y = x_all.loc[usable], y_all.loc[usable]
    dev = x.index < '2026-02-01'
    cal = (x.index >= '2026-02-01') & (x.index < '2026-04-30')
    embargo = (x.index >= '2026-04-30') & (x.index < '2026-05-01')
    test = x.index >= '2026-05-01'
    assert sum(dev)+sum(cal)+sum(embargo)+sum(test) == len(x)
    assert x.index[cal].max() <= x.index[test].min()-pd.Timedelta(days=2)
    splits = list(TimeSeriesSplit(n_splits=3, test_size=45, gap=1).split(x.loc[dev]))
    specs = regressors()
    cv_rows = []
    for name, spec in specs.items():
        for fold, (tr, va) in enumerate(splits, 1):
            xd, yd = x.loc[dev], y.loc[dev]
            model = fit_reg(name, spec, xd.iloc[tr], yd.iloc[tr])
            pred = reg_predict(name, model, xd.iloc[va])
            cv_rows.append({'model':name, 'fold':fold, 'train_end':str(xd.index[tr[-1]].date()),
                            'validation_start':str(xd.index[va[0]].date()),
                            'validation_end':str(xd.index[va[-1]].date()), **reg_metrics(yd.iloc[va], pred)})
        logging.info('CV regression completed: %s', name)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(RESULTS/'regression_cv.csv', index=False)
    ranking = cv.groupby('model').MAE.mean().sort_values()
    winner = ranking.index[0]
    model = fit_reg(winner, specs[winner], x.loc[dev], y.loc[dev])
    logging.info('Frozen regression winner: %s', winner)

    # Classification is independently selected on development folds. Every fold
    # learns its high-demand threshold from its own training observations.
    cls_specs = {
        'logistic': make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=3000, random_state=SEED)),
        'random_forest': make_pipeline(SimpleImputer(add_indicator=True), RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight='balanced', random_state=SEED, n_jobs=2)),
        'hist_gradient_boosting': make_pipeline(SimpleImputer(add_indicator=True), HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7, l2_regularization=10, early_stopping=False, random_state=SEED)),
    }
    cls_rows = []
    for name, spec in cls_specs.items():
        for fold, (tr, va) in enumerate(splits, 1):
            xd, yd = x.loc[dev], y.loc[dev]
            threshold = float(yd.iloc[tr].quantile(.75))
            mdl = clone(spec).fit(xd.iloc[tr], (yd.iloc[tr]>threshold).astype(int))
            prob = mdl.predict_proba(xd.iloc[va])[:,1]
            cls_rows.append({'model':name,'fold':fold,'threshold':threshold,**cls_metrics((yd.iloc[va]>threshold).astype(int),prob>=.5,prob)})
    cls_cv = pd.DataFrame(cls_rows)
    cls_cv.to_csv(RESULTS/'classification_cv.csv', index=False)
    cls_winner = cls_cv.groupby('model').macro_F1.mean().idxmax()
    threshold = float(y.loc[dev].quantile(.75))
    clf = clone(cls_specs[cls_winner]).fit(x.loc[dev], (y.loc[dev]>threshold).astype(int))

    # Calibration does not participate in model selection or fitting.
    cal_pred = reg_predict(winner, model, x.loc[cal])
    scores = np.abs(y.loc[cal].to_numpy()-cal_pred)/np.maximum(cal_pred,1)
    k = min(len(scores), math.ceil((len(scores)+1)*.90))
    q = float(np.sort(scores)[k-1])
    rows = []
    metrics = {}
    for part, mask in [('calibration', cal), ('test',test)]:
        pred = reg_predict(winner, model, x.loc[mask])
        prob = clf.predict_proba(x.loc[mask])[:,1]
        out = pd.DataFrame({'actual':y.loc[mask], 'predicted':pred,
            'lower90':np.maximum(0,pred*(1-q)), 'upper90':pred*(1+q),
            'seasonal_baseline':x.loc[mask].seasonal_baseline, 'last_available_baseline':x.loc[mask].last_known,
            'actual_high':(y.loc[mask]>threshold).astype(int), 'predicted_high':(prob>=.5).astype(int),
            'high_probability':prob, 'data_age_days':x.loc[mask].data_age_days,
            'past_count_28':x.loc[mask].past_count_28})
        out['absolute_error'] = abs(out.actual-out.predicted)
        out['percentage_error'] = out.absolute_error/out.actual*100
        out['covered90'] = (out.actual>=out.lower90)&(out.actual<=out.upper90)
        out['split'] = part
        rows.append(out)
        metrics[part] = {'regression':reg_metrics(out.actual,out.predicted),
            'seasonal_baseline':reg_metrics(out.actual,out.seasonal_baseline),
            'last_available_baseline':reg_metrics(out.actual,out.last_available_baseline),
            'classification':cls_metrics(out.actual_high,out.predicted_high,out.high_probability),
            'majority_baseline':cls_metrics(out.actual_high,np.zeros(len(out)),np.zeros(len(out))),
            'interval_coverage':float(out.covered90.mean()),
            'interval_mean_width':float((out.upper90-out.lower90).mean())}
    predictions = pd.concat(rows)
    predictions.to_csv(RESULTS/'predictions.csv',encoding='utf-8-sig')
    test_out = predictions.loc[predictions.split=='test'].copy()
    test_out.sort_values('absolute_error',ascending=False).head(15).to_csv(RESULTS/'bad_cases.csv',encoding='utf-8-sig')
    slices=[]
    slice_defs={'weekday':test_out.index.dayofweek<5,'weekend':test_out.index.dayofweek>=5,
                'high_demand':test_out.actual_high==1,'normal_demand':test_out.actual_high==0,
                'source_age_2d':test_out.data_age_days==2,'source_age_over_2d':test_out.data_age_days>2}
    for name, mask in slice_defs.items():
        part=test_out.loc[mask]
        if len(part):slices.append({'slice':name,**reg_metrics(part.actual,part.predicted),
                                   'coverage90':float(part.covered90.mean())})
    pd.DataFrame(slices).to_csv(RESULTS/'error_slices.csv',index=False)
    # Paired block bootstrap retains local residual dependence better than iid resampling.
    delta=(abs(test_out.actual-test_out.seasonal_baseline)-test_out.absolute_error).to_numpy()
    rng=np.random.default_rng(SEED); boot=[]; n=len(delta); block=7
    for _ in range(1500):
        starts=rng.integers(0,n,math.ceil(n/block)); ind=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        boot.append(float(delta[ind].mean()))
    metrics['paired_bootstrap']={'block_observations':7,'replicates':1500,
        'MAE_improvement_over_seasonal':float(delta.mean()),
        'percentile95_interval':np.quantile(boot,[.025,.975]).tolist(),
        'caveat':'Blocks are seven observed rows, not seven complete calendar days; descriptive uncertainty only.'}

    # Descriptive clustering is fitted on development dates; future dates use fixed centroids.
    cf=cluster_features(data); cd=cf.index<'2026-02-01'
    scaler=StandardScaler().fit(cf.loc[cd]); z=scaler.transform(cf)
    candidates=[]; kms={}
    for nc in range(2,6):
        km=KMeans(n_clusters=nc,n_init=20,random_state=SEED).fit(z[cd]);kms[nc]=km
        candidates.append({'algorithm':'KMeans','k':nc,'silhouette':float(silhouette_score(z[cd],km.labels_)),
                           'SSE':float(km.inertia_),'davies_bouldin':float(davies_bouldin_score(z[cd],km.labels_))})
    best_k=max(candidates,key=lambda t:t['silhouette'])['k'];km=kms[best_k]
    gmm=GaussianMixture(n_components=best_k,n_init=5,random_state=SEED).fit(z[cd])
    g_labels=gmm.predict(z[cd])
    candidates.append({'algorithm':'GMM','k':best_k,'silhouette':float(silhouette_score(z[cd],g_labels)),
                       'SSE':None,'davies_bouldin':float(davies_bouldin_score(z[cd],g_labels))})
    pd.DataFrame(candidates).to_csv(RESULTS/'clustering_selection.csv',index=False)
    labels=km.predict(z); clustering=data[['card_total','Total Smart Cards','Total NCMC','Total Tokens']].copy()
    clustering['cluster']=labels;clustering['split']=np.where(cd,'development','later')
    clustering.to_csv(RESULTS/'clusters.csv',encoding='utf-8-sig')
    profiles=clustering.groupby('cluster').agg(n=('card_total','size'),mean_card=('card_total','mean'),mean_ncmc=('Total NCMC','mean'),mean_tokens=('Total Tokens','mean'))
    profiles.to_csv(RESULTS/'cluster_profiles.csv')
    ari=[]
    for seed in [7,21,84]:
        alt=KMeans(n_clusters=best_k,n_init=20,random_state=seed).fit(z[cd]);ari.append(float(adjusted_rand_score(km.labels_,alt.labels_)))
    metrics['clustering']={'k':best_k,'development_silhouette':next(v['silhouette'] for v in candidates if v['algorithm']=='KMeans' and v['k']==best_k),
        'seed_stability_ARI':ari,'features':list(cf.columns),'purpose':'Descriptive fare-media day types, not stations or travel purpose labels'}

    # The stored evaluation model remains frozen; this forecast uses its same interval calibration.
    future_date=data.index.max()+pd.Timedelta(days=2)
    future_x,_=make_features(data,through=future_date)
    fx=future_x.loc[[future_date]]
    fp=float(reg_predict(winner,model,fx)[0]);fprob=float(clf.predict_proba(fx)[0,1])
    forecast={'date':str(future_date.date()),'information_cutoff':str(data.index.max().date()),
        'issue_date_assumption':str((future_date-pd.Timedelta(days=1)).date()),
        'prediction':fp,'lower90':max(0,fp*(1-q)),'upper90':fp*(1+q),
        'high_probability':fprob,'predicted_high':bool(fprob>=.5),'high_threshold':threshold,
        'status':'ARCHIVED_SNAPSHOT_FORECAST_NOT_A_CURRENT_LIVE_FORECAST',
        'model':winner,'scope':'Smart Cards + NCMC only; not all metro passengers',
        'interval_caveat':'Nominal 90%; time dependence and drift mean no guaranteed future coverage'}
    save_json(RESULTS/'next_forecast.json',forecast)
    pd.DataFrame([forecast]).to_csv(RESULTS/'next_forecast.csv',index=False,encoding='utf-8-sig')
    splitdata=x.copy();splitdata['target']=y;splitdata['split']=np.select([dev,cal,embargo,test],['development','calibration','embargo','test'],default='unassigned')
    splitdata.to_csv(ROOT/'data/processed/model_table.csv',encoding='utf-8-sig')
    save_json(RESULTS/'split_manifest.json',{part:{'n':int(sum(mask)),'start':str(x.index[mask][0].date()),'end':str(x.index[mask][-1].date())} for part,mask in [('development',dev),('calibration',cal),('test',test)]})
    joblib.dump({'regressor':model,'regressor_name':winner,'classifier':clf,'classifier_name':cls_winner,
                 'high_threshold':threshold,'relative_interval_radius':q,'features':list(x.columns),
                 'cluster_model':km,'cluster_scaler':scaler,'availability_lag_days':2,
                 'calibration_end':str(x.index[cal].max().date()),
                 'first_valid_prediction_date':str(x.index[test].min().date())},ROOT/'models/model_bundle.joblib')
    with sqlite3.connect(ROOT/'data/processed/traffic.sqlite') as con:
        data.reset_index().to_sql('daily_observations',con,if_exists='replace',index=False)
        splitdata.reset_index().to_sql('model_features',con,if_exists='replace',index=False)
        predictions.reset_index().to_sql('predictions',con,if_exists='replace',index=False)
        con.execute('CREATE UNIQUE INDEX IF NOT EXISTS observation_date ON daily_observations(date)')
        con.execute('CREATE UNIQUE INDEX IF NOT EXISTS prediction_date ON predictions(date)')
    metrics['selected']={'regression':winner,'classification':cls_winner,'high_threshold':threshold,
                         'relative_interval_radius':q,'selection':'Development time-series CV only'}
    metrics['runtime_seconds']=time.perf_counter()-started
    save_json(RESULTS/'metrics.json',metrics)
    save_json(RESULTS/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
        'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,
        'seed':SEED,'raw_sha256':audit['sha256']})
    logging.info('Finished in %.1fs; test %s',metrics['runtime_seconds'],metrics['test']['regression'])


if __name__ == '__main__':
    main()
