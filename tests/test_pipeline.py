"""Behavioral checks: calendar causality, invalid input, artifacts and reproducibility."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_pipeline import ROOT, RAW, load_clean, make_features
from predict import forecast_for_date


class TrafficTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.audit=load_clean()
        cls.x,cls.y=make_features(cls.data)

    def invalid_csv(self, mutate):
        raw=pd.read_csv(RAW); changed=mutate(raw)
        with tempfile.TemporaryDirectory(prefix='traffic-test-') as temp:
            p=Path(temp)/'case.csv';changed.to_csv(p,index=False)
            with self.assertRaises(ValueError):load_clean(p)

    def test_negative_counts_rejected(self):
        def mutate(d): d.loc[0,'Total NCMC']=-1;return d
        self.invalid_csv(mutate)

    def test_missing_schema_rejected(self):
        self.invalid_csv(lambda d:d.drop(columns=['Total NCMC']))

    def test_unknown_counts_not_zero(self):
        def mutate(d):d.loc[0,'Total NCMC']=np.nan;return d
        self.invalid_csv(mutate)

    def test_conflicting_dates_rejected(self):
        def mutate(d):r=d.iloc[[0]].copy();r['Total Tokens']+=1;return pd.concat([d,r])
        self.invalid_csv(mutate)

    def test_target_accounting(self):
        self.assertEqual(self.audit['smart_identity_mismatches'],0)
        np.testing.assert_array_equal(self.data.card_total,self.data['Total Smart Cards']+self.data['Total NCMC'])

    def test_missing_days_remain_unlabelled(self):
        self.assertEqual(int(self.y.isna().sum()),self.audit['missing_days'])
        self.assertGreater(self.audit['missing_days'],0)

    def test_lag_is_calendar_not_row(self):
        for date in self.x.index[28:]:
            prior=date-pd.Timedelta(days=7)
            expected=self.data.card_total.get(prior,np.nan)
            actual=self.x.at[date,'lag_7']
            self.assertTrue((np.isnan(expected) and np.isnan(actual)) or expected==actual)

    def test_future_and_unpublished_values_cannot_affect_features(self):
        date=pd.Timestamp('2026-06-01')
        altered=self.data.copy()
        altered.loc[altered.index>=date-pd.Timedelta(days=1),['card_total','Total Tokens','Total NCMC']]*=100
        xx,_=make_features(altered)
        np.testing.assert_allclose(self.x.loc[:date],xx.loc[:date],equal_nan=True)

    def test_prefix_features_equal_full_history(self):
        date=pd.Timestamp('2026-06-01')
        xx,_=make_features(self.data.loc[:date-pd.Timedelta(days=2)],through=date)
        np.testing.assert_allclose(self.x.loc[date],xx.loc[date],equal_nan=True)

    def test_outputs_finite_and_interval_ordered(self):
        p=pd.read_csv(ROOT/'results/predictions.csv')
        self.assertTrue(np.isfinite(p.select_dtypes('number')).all().all())
        self.assertTrue((p.lower90<=p.predicted).all() and (p.predicted<=p.upper90).all())
        self.assertTrue(p.high_probability.between(0,1).all())

    def test_split_disjoint_ordered(self):
        m=json.loads((ROOT/'results/split_manifest.json').read_text())
        self.assertLess(m['development']['end'],m['calibration']['start'])
        self.assertLess(m['calibration']['end'],m['test']['start'])

    def test_metrics_match_row_predictions(self):
        p=pd.read_csv(ROOT/'results/predictions.csv');p=p[p.split=='test']
        m=json.loads((ROOT/'results/metrics.json').read_text())['test']['regression']
        self.assertAlmostEqual(float(abs(p.actual-p.predicted).mean()),m['MAE'],places=7)

    def test_snapshot_forecast_reproduced(self):
        expected=json.loads((ROOT/'results/next_forecast.json').read_text())
        got=forecast_for_date(expected['date'])
        for key in ['prediction','lower90','upper90','high_probability']:
            self.assertAlmostEqual(got[key],expected[key],places=8)

    def test_stale_forecast_rejected(self):
        with self.assertRaisesRegex(ValueError,'Stale input'):
            forecast_for_date('2026-09-10')

    def test_too_little_history_rejected(self):
        with self.assertRaisesRegex(ValueError,'Insufficient history'):
            forecast_for_date('2024-11-01')

    def test_in_sample_replay_rejected(self):
        with self.assertRaisesRegex(ValueError,'out-of-sample'):
            forecast_for_date('2025-08-01')

    def test_calibration_period_replay_rejected(self):
        with self.assertRaisesRegex(ValueError,'calibration labels'):
            forecast_for_date('2026-03-01')

    def test_calibration_labels_available_before_first_test_forecast(self):
        p=pd.read_csv(ROOT/'results/predictions.csv',parse_dates=['date'])
        cal_end=p.loc[p.split=='calibration','date'].max()
        first_test=p.loc[p.split=='test','date'].min()
        self.assertLessEqual(cal_end,first_test-pd.Timedelta(days=2))


if __name__=='__main__':unittest.main(verbosity=2)
