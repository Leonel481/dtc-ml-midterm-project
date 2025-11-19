import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import yaml
import joblib
import pathlib
import os
from dotenv import load_dotenv

current_dir = pathlib.Path(__file__).parent.resolve()
project_root = current_dir.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

DATA_PATH_FILE = os.getenv('DATA_PATH_FILE')

class DataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        self.month_map = {
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
        
        self.day_map = {
                        'mon': 1,'tue': 2,'wed': 3,
                        'thu': 4,'fri': 5
                        }
        self.modes_ = {}

    def fit(self, X, y = None):

        cols_mode = ['job', 'marital', 'education', 'housing', 'loan']
        for col in cols_mode:
            if col in X.columns:
                self.modes_[col] = X[col].mode()[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df['month'] = df['month'].map(self.month_map)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['day_of_week'] = df['day_of_week'].map(self.day_map)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

        df['pdays'] = df['pdays'].replace(999, np.nan)
        df['pdays_missing'] = df['pdays'].isna().astype(int)
        df['pdays'] = df['pdays'].fillna(-1)

        df['campaign_log'] = np.log1p(df['campaign'])
        df['previous_log'] = np.log1p(df['previous'])

        df = df.drop(columns=['month','day_of_week','default', 'duration', 'emp.var.rate', 'euribor3m','previous','campaign'])

        return df

def create_pipeline() -> Pipeline:

    numeric_features = ['age', 'pdays', 'campaign_log', 'previous_log', 
                'cons.price.idx', 'cons.conf.idx', 'nr.employed']

    categorical_features = ['job', 'marital', 'education', 'housing', 
                            'loan', 'contact', 'poutcome']

    passthrough_features = ['month_sin','month_cos','day_sin','day_cos',
                            'pdays_missing']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('passthrough', 'passthrough', passthrough_features)
            ], 
            remainder='drop'
        ).set_output(transform="pandas")

    lgbm = LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1
            )

    pipeline = Pipeline(steps=[
        ('data_preprocessor', DataPreprocessor()),
        ('preprocessor', preprocessor),
        ('model', lgbm)
    ])

    return pipeline

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    df = pd.read_csv(DATA_PATH_FILE, sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    X = df.drop(columns=['y'])
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    neg, pos = y.value_counts()[0], y.value_counts()[1]
    scale_pos_weight = neg / pos

    pipeline_ml = create_pipeline()
    pipeline_ml.set_params(model__scale_pos_weight=scale_pos_weight)

    search_space_raw = instantiate(cfg.model.params)
    search_space_raw = OmegaConf.to_container(search_space_raw, resolve=True)
    param_distributions = {f'model__{k}': v for k, v in search_space_raw.items()}

    random_search = RandomizedSearchCV(
        estimator=pipeline_ml,
        param_distributions=param_distributions,
        scoring='recall',
        n_iter=50,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    random_search.fit(X_train, y_train)
    print(f'Best Score(CV Recall): {random_search.best_score_:.4f}')
    print(f"Bests Params: {random_search.best_params_}")

    test_score = random_search.score(X_test, y_test)
    print(f'Score Test Set (Holdout): {test_score:.4f}')


    output_path = project_root / 'model_production.joblib'
    joblib.dump(random_search.best_estimator_, output_path)
    print(f"Modelo guardado en: {output_path}")

if __name__ == "__main__":

    main()
