from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import TARGET_COL, COUNTRY_COL, RANDOM_STATE

def train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL])

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in df.columns if c not in (TARGET_COL, COUNTRY_COL)]
    cat_cols = [COUNTRY_COL]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
