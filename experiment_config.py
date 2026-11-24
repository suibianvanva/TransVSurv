import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler


TEMPORAL_FEATURES = ['SBP', 'DBP', 'HR', 'BMI', 'LDL', 'TG', 'HDL', 'TC', 'SCR', 'BUN', 'UA', 'GLU']
STATIC_FEATURES = ['SEX', 'Education', 'Marital', 'Smoking', 'Occupation']

N_EVENTS = 2
EVENT_CODES = [1, 2]
LOOKBACK = 5
PRED_TIME = 5
EVAL_TIMES = [1, 3, 5]

WEITANG_PATH = 'Data/相城训练数据.csv'
KUNSHAN_PATH = 'Data/昆山外部验证数据.csv'


def load_aligned_data():
    wt = pd.read_csv(WEITANG_PATH)
    ks = pd.read_csv(KUNSHAN_PATH)
    for c in TEMPORAL_FEATURES + STATIC_FEATURES:
        if c in wt.columns:
            wt[c] = pd.to_numeric(wt[c], errors='coerce')
        if c in ks.columns:
            ks[c] = pd.to_numeric(ks[c], errors='coerce')

    meta_cols = ['ID', 'time_idx']
    label_cols = ['event', 'final_event', 'time']

    def align(df):
        for c in TEMPORAL_FEATURES + STATIC_FEATURES:
            if c not in df.columns:
                df[c] = np.nan
        cols = meta_cols + STATIC_FEATURES + TEMPORAL_FEATURES + [c for c in label_cols if c in df.columns]
        return df[cols].copy()

    return align(wt), align(ks)


def fit_temporal_scaler(df):
    scaler = StandardScaler()

    scaler.fit(df[TEMPORAL_FEATURES].to_numpy(dtype=np.float64))
    return scaler


def apply_temporal_scaler(df, scaler):
    df = df.copy()
    arr = df[TEMPORAL_FEATURES].to_numpy(dtype=np.float64)
    scaled = scaler.transform(arr)
    df[TEMPORAL_FEATURES] = scaled
    return df


def patient_stratified_split(df, test_size=0.2, seed=1234):
    pat = df.groupby('ID')['final_event'].first().reset_index()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(sss.split(pat['ID'], pat['final_event']))
    return pat['ID'].iloc[rest_idx].values, pat['ID'].iloc[test_idx].values


def patient_stratified_kfold(df, ids, n_splits=5, seed=1234):
    pat = df[df['ID'].isin(ids)].groupby('ID')['final_event'].first()
    pat = pat.reindex(ids)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in skf.split(pat.index, pat.values):
        folds.append((pat.index[tr_idx].values, pat.index[va_idx].values))
    return folds


def subset_by_ids(df, ids):
    return df[df['ID'].isin(ids)].copy()
