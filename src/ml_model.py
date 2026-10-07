
#  USING LINEAR REGESSION MODEL

import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv(
    "food_waste_management_dataset_cleaned.csv"
)

print("Dataset loaded:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "expected_students",
    "menu",
    "day_of_week",
    "weather",
    "temperature_c",
    "is_holiday",
    "is_exam_day",
    "special_event"
]

target = "food_consumed_portions"

X = df[features]
y = df[target]


# ==========================================
# 3. IDENTIFY COLUMN TYPES
# ==========================================

categorical_features = [
    "menu",
    "day_of_week",
    "weather"
]

numerical_features = [
    "expected_students",
    "temperature_c",
    "is_holiday",
    "is_exam_day",
    "special_event"
]


# ==========================================
# 4. PREPROCESSING
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
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# 5. CREATE MODEL PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n===== LINEAR REGRESSION RESULTS =====")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 3))

# ==========================================
# CROSS VALIDATION
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
)

cv_mae = -cv_scores

print("\n===== 5-FOLD CROSS VALIDATION =====")

print("MAE for each fold:",
      cv_mae.round(2))

print("Average MAE:",
      round(cv_mae.mean(), 2))

print("MAE standard deviation:",
      round(cv_mae.std(), 2))

# ==========================================
# 10. SHOW ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(2)
})

print("\n===== SAMPLE PREDICTIONS =====")
print(results)