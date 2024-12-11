#%%

import pandas as pd
from pathlib import Path
import holidays
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from time import time

# Function to encode dates
def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    return X

# Function to preprocess data
def preprocess_data(data):
    data = data.copy()
    data = _encode_dates(data)
    data["weekend"] = (data["weekday"] > 4).astype(int)
    FR_holidays = holidays.FR(years=range(2019, 2022))
    data["FR_holidays"] = data["date"].dt.date.isin(FR_holidays).astype(int)
    return data

#%% Load and preprocess train data
#data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet") # to load on Kaggle 
data = pd.read_parquet(Path("data") / "train.parquet") # to load locally
train_data = preprocess_data(data)

weather_data = pd.read_csv(Path("data") / "external_data.csv")
weather_data["date"] = pd.to_datetime(weather_data["date"], errors="coerce")
weather_data = _encode_dates(weather_data)
weather_data = weather_data.drop_duplicates(subset="date")
weather_data.set_index("date", inplace=True)
weather_data = weather_data.resample("H").interpolate(method="linear")
weather_data.reset_index(inplace=True)

merged_data = pd.merge(train_data, weather_data, on="date", how="left")
merged_data = merged_data.loc[:, ~merged_data.columns.str.endswith(("_x", "_y"))]

def get_train_data(data=merged_data, target_column="log_bike_count"):
    data = data.sort_values(["date", "counter_name"])
    y_array = data[target_column].values
    # Keep the `bike_count` column for filtering in train-test split
    X_df = data.drop([target_column], axis=1)
    return X_df, y_array

X, y = get_train_data(merged_data)

# Split train and validation data
def train_test_split_temporal(X, y, delta_threshold="30 days"):
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date

    # Split train and validation data
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    # Drop rows where bike_count == 0 in the training data
    train_mask = X_train["bike_count"] != 0
    X_train = X_train.loc[train_mask]
    y_train = y_train[train_mask]

    # Drop `bike_count` from training and validation features after filtering
    X_train = X_train.drop(["bike_count"], axis=1)
    X_valid = X_valid.drop(["bike_count"], axis=1)

    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

# Best model without weather data
# 0.72 for  depth = 10, n = 100, lr = 0.3


#%% Define preprocessing pipeline
date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()
#date_cols = [col for col in date_cols if col != "day"]

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]

numerical_encoder = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)
numerical_corr_cols = ["u", "t", "tx12", "tn12", "rafper", "td", "raf10", "ff", "nnuage3", "vv"]

binary_cols = ["weekend", "FR_holidays"]

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        #("num", numerical_encoder, numerical_corr_cols),
        ("binary", "passthrough", binary_cols)
    ]
)


# 2. XGB Regressor
# With weather data

regressor = XGBRegressor(random_state=42, max_depth=10, n_estimators=200, learning_rate=0.3)

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        #("num", numerical_encoder, numerical_corr_cols),
        ("binary", "passthrough", binary_cols)
    ]
)

#%%
start = time()
pipe = make_pipeline(date_encoder, preprocessor, regressor)
pipe.fit(X_train, y_train)
elapsed_time = time() - start
print(f"Training time for GradientBoosting with weather: {elapsed_time:.2f} seconds")
print(f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}")
print(f"Valid set, RMSE={mean_squared_error(y_valid, pipe.predict(X_valid), squared=False):.2f}")

# Save predictions

#test_data = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet") # to load on Kaggle
test_data = pd.read_parquet(Path("data") / "final_test.parquet") # to load locally
test_data = preprocess_data(test_data)

merged_data = pd.merge(test_data, weather_data, on="date", how="left") # merge test and weather data
merged_data = merged_data.loc[:, ~merged_data.columns.str.endswith(("_x", "_y"))] # Drop redundant date columns 


X_test = merged_data
y_pred = pipe.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
output_path = "submission_XGB_without_weather_zeros_dropped,d=10,n=100,l=0.3_with_31days.csv"
results.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

#%%

