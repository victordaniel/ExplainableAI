"""Dataset loaders. Each returns (X, y, A, cat_cols, num_cols):
X  : DataFrame of original variables (categoricals as strings),
y  : 1 = favourable outcome,
A  : 1 = unprivileged group,
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'

ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
              'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
              'hours-per-week', 'native-country', 'income']


def adult():
    a = pd.read_csv(DATA / 'adult.data', header=None, names=ADULT_COLS, skipinitialspace=True)
    b = pd.read_csv(DATA / 'adult.test', header=None, names=ADULT_COLS, skipinitialspace=True, skiprows=1)
    df = pd.concat([a, b], ignore_index=True)
    df = df.replace('?', np.nan).dropna()
    y = df['income'].str.replace('.', '', regex=False).eq('>50K').astype(int).to_numpy()
    A = df['sex'].eq('Female').astype(int).to_numpy()
    X = df.drop(columns=['income', 'fnlwgt', 'education']).reset_index(drop=True)
    cat = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    num = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    return X, y, A, cat, num, 'sex'


def acs_income(n=20000, seed=2026):
    from folktables import ACSDataSource, ACSIncome
    ds = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir=str(DATA))
    raw = ds.get_data(states=['CA'], download=True)      # uses the local copy if already present
    Xf, yf, _ = ACSIncome.df_to_pandas(raw)
    yv = yf.iloc[:, 0].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    idx = []
    for c in (0, 1):                      # stratified subsample
        ic = np.flatnonzero(yv == c)
        k = int(round(n * len(ic) / len(yv)))
        idx.extend(rng.choice(ic, k, replace=False))
    idx = np.sort(np.array(idx))
    d = Xf.iloc[idx].reset_index(drop=True)
    X = pd.DataFrame({
        'AGEP': d['AGEP'].astype(float), 'SCHL': d['SCHL'].astype(float), 'WKHP': d['WKHP'].astype(float),
        'COW': d['COW'].astype(int).astype(str), 'MAR': d['MAR'].astype(int).astype(str),
        'RELP': d['RELP'].astype(int).astype(str), 'RAC1P': d['RAC1P'].astype(int).astype(str),
        'OCCP': (d['OCCP'].astype(int) // 1000).astype(str),
        'POBP': np.where(d['POBP'].astype(int) < 100, 'US', 'nonUS'),
        'SEX': np.where(d['SEX'].astype(int) == 2, 'F', 'M'),
    })
    y = yv[idx]
    A = (X['SEX'] == 'F').astype(int).to_numpy()
    cat = ['COW', 'MAR', 'RELP', 'RAC1P', 'OCCP', 'POBP', 'SEX']
    num = ['AGEP', 'SCHL', 'WKHP']
    return X, y, A, cat, num, 'SEX'


def compas():
    df = pd.read_csv(DATA / 'compas-scores-two-years.csv')
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1) & (df.c_charge_degree != 'O') & (df.score_text != 'N/A')]
    df = df[df.race.isin(['African-American', 'Caucasian'])].reset_index(drop=True)
    y = (df.two_year_recid == 0).astype(int).to_numpy()          # favourable: no recidivism
    A = (df.race == 'African-American').astype(int).to_numpy()
    num = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
    cat = ['sex', 'race', 'c_charge_degree']
    X = df[num + cat].copy()
    for c in num:
        X[c] = X[c].astype(float)
    return X, y, A, cat, num, 'race'


def german():
    cols = [f'A{i}' for i in range(1, 21)] + ['target']
    df = pd.read_csv(DATA / 'german.data', sep=' ', header=None, names=cols)
    y = (df.target == 1).astype(int).to_numpy()                   # favourable: good credit
    num_i = [2, 5, 8, 11, 13, 16, 18]
    num = [f'A{i}' for i in num_i]
    cat = [f'A{i}' for i in range(1, 21) if i not in num_i]
    X = df.drop(columns='target').copy()
    for c in num:
        X[c] = X[c].astype(float)
    A = (X['A13'] <= 25).astype(int).to_numpy()                   # A13 = age in years
    X = X.rename(columns={'A13': 'age'})
    num = ['age' if c == 'A13' else c for c in num]
    return X, y, A, cat, num, 'age'


LOADERS = {'adult': adult, 'acs_income': acs_income, 'compas': compas, 'german': german}
