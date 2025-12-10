#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------
# 1. LOAD DATA
# -------------------------------------
df = pd.read_csv(r"C:\Users\srini\Downloads\house_price_sample.csv")

print("First 5 rows:")
print(df.head(), "\n")

print("Columns in dataset:")
print(df.columns, "\n")

# -------------------------------------
# 2. PREPROCESSING
# -------------------------------------

# Split features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Identify numeric & categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ["ocean_proximity"]

# Preprocess numeric data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Preprocess categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing pipelines
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# -------------------------------------
# 3. CREATE MODEL PIPELINE
# -------------------------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# -------------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# 5. TRAIN THE MODEL
# -------------------------------------
model.fit(X_train, y_train)

# -------------------------------------
# 6. PREDICT
# -------------------------------------
predictions = model.predict(X_test)

# -------------------------------------
# 7. EVALUATE
# -------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Model Performance:")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}\n")

# -------------------------------------
# 8. EXAMPLE PREDICTION
# -------------------------------------
example = X_test.iloc[0:1]
pred_value = model.predict(example)[0]

print("Example Input (first row of test data):")
print(example, "\n")

print(f"Predicted House Value: ${pred_value:,.2f}")


# In[10]:


print(df.columns)


# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------
# 1. LOAD DATA
# -------------------------------------
df = pd.read_csv(r"C:\Users\srini\Downloads\house_price_sample.csv")

print("Dataset Preview:")
print(df.head(), "\n")
print("Columns:", df.columns, "\n")

# -------------------------------------
# 2. SPLIT FEATURES & TARGET
# -------------------------------------
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Identify numeric columns (all are numeric here)
numeric_cols = X.columns

# -------------------------------------
# 3. PREPROCESSING PIPELINE
# -------------------------------------
preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -------------------------------------
# 4. MODEL PIPELINE
# -------------------------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# -------------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# 6. TRAIN MODEL
# -------------------------------------
model.fit(X_train, y_train)

# -------------------------------------
# 7. PREDICTION & EVALUATION
# -------------------------------------
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n📊 Model Performance:")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# -------------------------------------
# 8. SAMPLE PREDICTION
# -------------------------------------
example = X_test.iloc[:1]
print("\nSample Input:")
print(example)

print("\nPredicted House Price:")
print(model.predict(example))


# In[ ]:




