"""CLI inference with explicit availability cutoff and stale-input rejection."""
import argparse
import json
import joblib
import pandas as pd
from data_pipeline import ROOT, RAW, load_clean, make_features
from experiment import reg_predict


def forecast_for_date(date, raw=RAW):
    date = pd.Timestamp(date)
    if pd.isna(date) or date != date.normalize():
        raise ValueError('Use a calendar date YYYY-MM-DD')
    data, _ = load_clean(raw)
    bundle = joblib.load(ROOT/'models/model_bundle.joblib')
    cutoff = date-pd.Timedelta(days=bundle['availability_lag_days'])
    history = data.loc[data.index <= cutoff]
    if len(history)<28 or (cutoff-history.index.min()).days<28:
        raise ValueError('Insufficient history; need at least 28 observations and 28 calendar days')
    if date < pd.Timestamp(bundle['first_valid_prediction_date']):
        raise ValueError('Full model and interval are out-of-sample only from '+bundle['first_valid_prediction_date']+'; calibration labels must be available before issue time')
    age=int((date-history.index.max()).days)
    if age>7:
        raise ValueError(f'Stale input: latest usable observation is {age} days before prediction; update data first')
    x,_=make_features(history,through=date)
    row=x.loc[[date],bundle['features']]
    point=float(reg_predict(bundle['regressor_name'],bundle['regressor'],row)[0])
    prob=float(bundle['classifier'].predict_proba(row)[0,1]);q=bundle['relative_interval_radius']
    return {'date':str(date.date()),'information_cutoff':str(history.index.max().date()),
            'prediction':point,'lower90':max(0,point*(1-q)),'upper90':point*(1+q),
            'high_probability':prob,'predicted_high':bool(prob>=.5),
            'high_threshold':bundle['high_threshold'],'data_age_days':age,
            'scope':'Total Smart Cards + Total NCMC, passenger trips/day',
            'status':'Historical replay' if date<=data.index.max() else 'Snapshot extension',
            'model_fitted_through':'2026-01-31',
            'interval_calibrated_through':bundle['calibration_end'],
            'warning':'No holiday/event features. Nominal interval is not a future coverage guarantee.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',required=True)
    parser.add_argument('--raw',default=str(RAW))
    args=parser.parse_args()
    try:
        result=forecast_for_date(args.date,args.raw)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,f'Cannot predict: {exc}\n')
