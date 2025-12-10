#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Load train.xlsx
# -----------------------------
df = pd.read_csv(r"Downloads/archive (1)/train.csv")

print(df.head())

# -----------------------------
# 2. Filter for one store and one department
# -----------------------------
# (Time series cannot handle multiple stores/departments together)
store_id = 1
dept_id = 1

data = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)].copy()

# -----------------------------
# 3. Process the date and sort
# -----------------------------
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.set_index('Date', inplace=True)

# Use only the sales column
ts = data['Weekly_Sales']

plt.figure(figsize=(12,5))
plt.plot(ts)
plt.title("Store 1 - Dept 1 Sales Over Time")
plt.show()

# -----------------------------
# 4. Train-test split
# -----------------------------
train_size = int(len(ts) * 0.8)
train = ts[:train_size]
test = ts[train_size:]

# -----------------------------
# 5. Build ARIMA Model
# -----------------------------
model = ARIMA(train, order=(2,1,2))
fit = model.fit()

# -----------------------------
# 6. Forecast
# -----------------------------
forecast = fit.forecast(steps=len(test))

# -----------------------------
# 7. Evaluate
# -----------------------------
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("MAE:", mae)
print("RMSE:", rmse)

# -----------------------------
# 8. Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Predicted')
plt.legend()
plt.title("Actual vs Forecasted Sales")
plt.show()

# -----------------------------
# 9. Predict future 30 weeks
# -----------------------------
future_model = ARIMA(ts, order=(2,1,2))
future_fit = future_model.fit()

future_forecast = future_fit.forecast(steps=30)

print("\nFuture 30 Weeks Forecast:")
print(future_forecast)

