import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load CSV file.
csv_file = r"C:\Users\Gabri\Downloads\NVDA.csv"
df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

# Filter data for the last 5 years.
df = df[df.index >= (df.index.max() - pd.DateOffset(years=5))]

# Create moving averages and forward returns
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# 5D = week, 20D = month, 260D = year.
df['5D_Return'] = df['Close'].shift(-5) / df['Close'] - 1
df['20D_Return'] = df['Close'].shift(-20) / df['Close'] - 1
df['260D_Return'] = df['Close'].shift(-260) / df['Close'] - 1

# Drop rows without values.
df.dropna(inplace=True)

# Select features and target.
features = ['Close', 'MA10', 'MA50', 'MA200']
target = '5D_Return'  # Change this to show different periods uch as 20D or 260D

X = df[features]
y = df[target]

# Scale the features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVR model.
svr_model = SVR(kernel='rbf', C=1000, gamma=0.1)
svr_model.fit(X_train, y_train)

# Predict the test set.
y_pred = svr_model.predict(X_test)

# Calculate RMSE and R-squared.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

# Predict next 30 days.
forecast_days = 30
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
future_features = df[features].iloc[-forecast_days:].values
future_scaled_features = scaler.transform(future_features)
future_predictions = svr_model.predict(future_scaled_features)

# Plot the 5 years of data with predicted 30 day returns.
plt.figure(figsize=(14, 7))
plt.plot(df.index, y, label='Actual Returns')
plt.plot(df.index[-len(y_test):], y_pred, label='Predicted Returns', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Actual vs Predicted Returns (Full 5 Years)')
plt.legend()
plt.show()

# Plot the next 30 days wrth of predictions.
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions, label='Future Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Predicted Returns')
plt.title('Predicted Returns for the Next 30 Days')
plt.legend()
plt.show()



