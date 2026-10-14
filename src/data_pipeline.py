"""Auditable daily data preparation. Missing calendar dates are never fabricated labels."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/NammaMetro_Ridership_Dataset.csv'
CARD_PARTS = ['Stored Value Card', 'One Day Pass', 'Three Day Pass', 'Five Day Pass']
QR_PARTS = ['QR NammaMetro', 'QR WhatsApp', 'QR Paytm', 'QR ONDC']
REQUIRED = ['Record Date', 'Total Smart Cards', *CARD_PARTS, 'Total Tokens',
            'Total NCMC', 'Group Ticket', 'Total QR', *QR_PARTS]


def load_clean(path=RAW):
    raw = pd.read_csv(path)
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    data = raw.copy()
    data['date'] = pd.to_datetime(data['Record Date'], format='%d-%m-%Y', errors='raise')
    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates()
    if data.date.duplicated().any():
        raise ValueError('Conflicting duplicate dates; reconcile against source before training')
    for col in REQUIRED[1:]:
        data[col] = pd.to_numeric(data[col], errors='raise')
        if data[col].isna().any() or not np.isfinite(data[col]).all():
            raise ValueError(f'Non-finite count in {col}; do not turn missing counts into zero')
        if (data[col] < 0).any() or (data[col] % 1 != 0).any():
            raise ValueError(f'Invalid passenger count in {col}')
    data = data.sort_values('date').set_index('date')
    data['smart_gap'] = data[CARD_PARTS].sum(axis=1) - data['Total Smart Cards']
    data['qr_gap'] = data[QR_PARTS].sum(axis=1) - data['Total QR']
    data['qr_gap_without_ondc'] = data[QR_PARTS[:-1]].sum(axis=1) - data['Total QR']
    data['card_total'] = data['Total Smart Cards'] + data['Total NCMC']
    if (data.smart_gap != 0).any() or (data.card_total <= 0).any():
        raise ValueError('Target failed accounting checks; manual source verification required')
    calendar = pd.date_range(data.index.min(), data.index.max(), freq='D')
    audit = {
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'rows': len(data), 'raw_columns': len(raw.columns),
        'exact_duplicates_removed': exact_duplicates,
        'start': str(data.index.min().date()), 'end': str(data.index.max().date()),
        'calendar_days': len(calendar), 'missing_days': len(calendar.difference(data.index)),
        'missing_dates': [str(d.date()) for d in calendar.difference(data.index)],
        'max_observation_gap_days': int(data.index.to_series().diff().dt.days.max()),
        'smart_identity_mismatches': int((data.smart_gap != 0).sum()),
        'qr_identity_mismatches': int((data.qr_gap != 0).sum()),
        'qr_matches_only_without_ondc': int(((data.qr_gap != 0) & (data.qr_gap_without_ondc == 0)).sum()),
        'zero_counts_by_column': {c: int((data[c] == 0).sum()) for c in REQUIRED[1:]},
        'target': 'Total Smart Cards + Total NCMC',
        'target_unit': 'recorded passenger trips per day for these card media only',
        'provenance_limit': 'Public operator data collected by author; historical rows not individually verified against operator archives',
        'qr_policy': 'Preserve and flag; exclude all QR fields from main forecasting and clustering; no guessed repairs',
    }
    return data, audit


def make_features(data, through=None, availability_lag=2):
    """At forecast date d, only dates <= d-2 are available by design.

    Calendar reindexing precedes all shifts. A missing day remains NaN;
    causal summaries may use earlier observations but targets are never imputed.
    """
    end = pd.Timestamp(through) if through is not None else data.index.max()
    idx = pd.date_range(data.index.min(), end, freq='D', name='date')
    daily = data.reindex(idx)
    y = daily['card_total']
    x = pd.DataFrame(index=idx)
    x['trend_days'] = (idx - pd.Timestamp('2024-01-01')).days
    for day in range(7):
        x[f'weekday_{day}'] = (idx.dayofweek == day).astype(int)
    x['annual_sin'] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    x['annual_cos'] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    for lag in [2, 3, 7, 14, 21, 28]:
        if lag >= availability_lag:
            x[f'lag_{lag}'] = y.shift(lag)
    available = y.shift(availability_lag)
    x['last_known'] = available.ffill()
    known_dates = pd.Series(idx, index=idx).where(y.notna()).shift(availability_lag).ffill()
    x['data_age_days'] = (pd.Series(idx, index=idx) - known_dates).dt.days
    for days in [7, 28]:
        roll = available.rolling(days, min_periods=1)
        x[f'past_mean_{days}'] = roll.mean()
        x[f'past_std_{days}'] = roll.std()
        x[f'past_count_{days}'] = roll.count()
    same_weekdays = pd.concat([y.shift(k) for k in [7, 14, 21, 28]], axis=1)
    x['seasonal_baseline'] = same_weekdays.mean(axis=1).fillna(x['past_mean_28']).fillna(x['last_known'])
    x['ncmc_share_last'] = (daily['Total NCMC'] / y).shift(availability_lag).ffill()
    x['tokens_last'] = daily['Total Tokens'].shift(availability_lag).ffill()
    return x.astype(float), y


def cluster_features(data):
    """Descriptive daily modes, NOT advance forecasts or station clusters."""
    return pd.DataFrame({
        'log_card_total': np.log1p(data.card_total),
        'ncmc_share': data['Total NCMC'] / data.card_total,
        'token_to_card_ratio': data['Total Tokens'] / data.card_total,
    }, index=data.index)
