#%%

import pandas as pd
from pathlib import Path
from joblib import load
import holidays
import numpy as np
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from time import time

# Load train data
#data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet") # to load on Kaggle 
data = pd.read_parquet(Path("data") / "train.parquet") # to load locally

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    return X

# Feature manipulation for train and test data
def preprocess_data(data):
    data = data.copy()

    # Apply date encoding
    data = _encode_dates(data)

    # Add weekend column
    data["weekend"] = (data["weekday"] > 4).astype(int)  # 1 stands for weekend, 0 stands for no weekend

    # Add French holidays column
    FR_holidays = holidays.FR(years=range(2019, 2022))
    data["FR_holidays"] = data["date"].dt.date.isin(FR_holidays).astype(int)

    return data

# Manipulate train data
train_data = preprocess_data(data)

# Add weather data
#weather_data = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv") # to load on Kaggle 
weather_data = pd.read_csv(Path("data") / "external_data.csv") # to load locally

weather_data["date"] = pd.to_datetime(weather_data["date"], errors="coerce")
weather_data = _encode_dates(weather_data)
weather_data = weather_data.drop_duplicates(subset="date") # Drop duplicate rows based on the 'date' column

# Interpolate linearly to get from 3 hour data to 1 hour data
weather_data.set_index("date", inplace=True)  # Set date as the index
weather_data = weather_data.resample("H").interpolate(method="linear")  # Interpolate missing values
weather_data.reset_index(inplace=True)  # Reset index

# Merge bike data with weather data using a left join
merged_data = pd.merge(train_data, weather_data, on="date", how="left")

# Drop redundant date columns to avoid duplicates in the final dataset
merged_data = merged_data.loc[:, ~merged_data.columns.str.endswith(("_x", "_y"))]

# Extract train and validate data
def get_train_data(data = merged_data, _target_column_name = "log_bike_count"):
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

X, y = get_train_data()

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

# Training the model
date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()
date_cols = [col for col in date_cols if col != "day"] # exclude "day" in one hot encoding

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]

numerical_encoder = make_pipeline(
    SimpleImputer(strategy="mean"),  # Replace NaNs of weather data with the mean 
    StandardScaler())
numerical_corr_cols = ["u", "t", "tx12", "tn12", "rafper", "td", "raf10", "ff", "nnuage3", "vv"] # Weather columns with correlation >|0.1| (see EDA)


binary_cols = ["weekend", "FR_holidays"] # No transformation required, they are already binary


preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
        #("num", numerical_encoder, numerical_corr_cols), # for this model, weather data does not improve
        ("binary", "passthrough", binary_cols)
    ]
)

regressor = RandomForestRegressor(random_state=42, max_depth=36, n_estimators=7, n_jobs=-1)
print(X_train.columns)
print(numerical_corr_cols)
#%%
start = time()
pipe = make_pipeline(date_encoder, preprocessor, regressor)
pipe.fit(X_train, y_train)
elapsed_time = time() - start
print(f"Training time for random forest: {elapsed_time:.2f} seconds")
print(f"Train set, RMSE={mean_squared_error(y_train, pipe.predict(X_train), squared=False):.2f}")
print(f"Valid set, RMSE={mean_squared_error(y_valid, pipe.predict(X_valid), squared=False):.2f}")


# Load the test data
#test_data = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet") # to load on Kaggle
test_data = pd.read_parquet(Path("data") / "final_test.parquet") # to load locally

test_data = preprocess_data(test_data)

# Merge test bike data with weather data using a left join
merged_data = pd.merge(test_data, weather_data, on="date", how="left")

# Drop redundant date columns to avoid duplicates in the final dataset
merged_data = merged_data.loc[:, ~merged_data.columns.str.endswith(("_x", "_y"))]

# Extract features (X) for prediction
X_test = merged_data

# Predict on test data
y_pred = pipe.predict(X_test)

# Save predictions
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
output_path = "submission.csv"
results.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")





# %%
