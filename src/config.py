EXPERIMENT_NAME = "Heart_Disease_Prediction"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TARGET_COLUMN = "Heart Disease"

NUMERIC_FEATURES = [
    "Age", "BP", "Cholesterol",
    "Max HR", "ST depression"
]

BINARY_FEATURES = [
    "Sex", "FBS over 120", "Exercise angina"
]

CATEGORICAL_FEATURES = [
    "Chest pain type",
    "EKG results",
    "Slope of ST",
    "Thallium"
]

ORDINAL_FEATURES = [
    "Number of vessels fluro"
]

DROP_COLUMNS = ["id"]
