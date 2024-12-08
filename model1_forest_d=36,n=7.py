import pandas as pd
from pathlib import Path
from joblib import load
from encode_dates import _encode_dates
import holidays
import numpy as np

# Load the trained model
model_filename = "model1_forest_d=36,n=7.joblib"

pipe = load(model_filename)

print(f"Loaded model from {model_filename}")

# Load the test data
test_data_path = Path("data") / "final_test.parquet"
test_data = pd.read_parquet(test_data_path)
print("Test data loaded.")

# Feature manipulation for test data
def preprocess_test_data(test_data):
    test_data = test_data.copy()

    # Apply date encoding
    test_data = _encode_dates(test_data)

    # Add weekend column
    test_data["weekend"] = (test_data["weekday"] > 4).astype(int)  # 1 stands for weekend, 0 stands for no weekend

    # Add French holidays column
    FR_holidays = holidays.FR(years=range(2019, 2022))
    test_data["FR_holidays"] = test_data["date"].dt.date.isin(FR_holidays).astype(int)

    return test_data

test_data = preprocess_test_data(test_data)
print("Test data preprocessed.")

# Extract features (X) for prediction
X_test = test_data

# Predict on test data
y_pred = pipe.predict(X_test)

# Save predictions
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
output_path = "submission1_Forest,d=36,n=7.csv"
results.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
