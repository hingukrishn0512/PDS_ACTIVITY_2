import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

DATA_PATH = "food_waste_management_dataset_cleaned.csv"

df = pd.read_csv(DATA_PATH)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

FEATURES = [
    "expected_students",
    "menu",
    "day_of_week",
    "weather",
    "temperature_c",
    "is_holiday",
    "is_exam_day",
    "special_event"
]

TARGET = "food_consumed_portions"


# ==========================================
# 3. COLUMN TYPES
# ==========================================

CATEGORICAL_FEATURES = [
    "menu",
    "day_of_week",
    "weather"
]

NUMERICAL_FEATURES = [
    "expected_students",
    "temperature_c",
    "is_holiday",
    "is_exam_day",
    "special_event"
]


# ==========================================
# 4. PREPROCESSOR
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ]
)


# ==========================================
# 5. MODEL
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

X = df[FEATURES]
y = df[TARGET]

model.fit(X, y)


# ==========================================
# 7. PREDICTION FUNCTION
# ==========================================

def predict_food_consumption(
    expected_students,
    menu,
    day_of_week,
    weather,
    temperature_c,
    is_holiday,
    is_exam_day,
    special_event
):
    input_data = pd.DataFrame([{
        "expected_students": expected_students,
        "menu": menu,
        "day_of_week": day_of_week,
        "weather": weather,
        "temperature_c": temperature_c,
        "is_holiday": is_holiday,
        "is_exam_day": is_exam_day,
        "special_event": special_event
    }])

    prediction = model.predict(input_data)[0]

    # Consumption cannot be negative
    prediction = max(0, prediction)

    # Round to nearest whole portion
    return round(prediction)