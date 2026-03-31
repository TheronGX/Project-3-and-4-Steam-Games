import pandas as pd
import numpy as np

df = pd.read_csv("steam_test.csv")

print(df.shape)
print(df.head())
print(df.info())

missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))
#-----------------Class-2-----------------#
target_col = "negative_ratings"

#Seperate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
#remove target from numeric predictors
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

# fill numeric missing values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# fill categorical missing values with mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# define X and y
X = df.drop("negative_ratings", axis=1)
y = df["negative_ratings"]

# scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#-----------------Class-3-----------------#
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

explained_variance = S**2 / np.sum(S**2)
cumulative_variance = np.cumsum(explained_variance)

import matplotlib.pyplot as plt

plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs Components")
plt.grid()
plt.show()

# Choose k to retain at least 90% variance
k = np.argmax(cumulative_variance >= 0.90) + 1

X_reduced = U[:, :k] @ np.diag(S[:k])
#-----------------Class-4-----------------#
from sklearn.linear_model import LinearRegression

# Train/test split first
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Evaluate scikit-learn LinearRegression on the test set
y_pred_lr = model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
#print(f"Linear Regression (sklearn) RMSE = {rmse_lr}")
#print(f"Linear Regression (sklearn) R^2 = {r2_lr}")
#print(f"Linear Regression (sklearn) MAE = {mae_lr}")


# Normal Equation using pseudo-inversion
X_design_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
beta = np.linalg.pinv(X_design_train.T @ X_design_train) @ X_design_train.T @ y_train
# Condition number for the 20% test set (include intercept)
X_design_test = np.column_stack((np.ones(X_test.shape[0]), X_test))
cond_number_test = np.linalg.cond(X_design_test)
print(f"Condition number (test set design matrix): {cond_number_test}")

# Predictions from the normal equation and basic evaluation/plotting
y_pred_normal = X_design_test @ beta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

rmse_ne = np.sqrt(mean_squared_error(y_test, y_pred_normal))
r2_ne = r2_score(y_test, y_pred_normal)
mae_ne = mean_absolute_error(y_test, y_pred_normal)
#print(f"Normal Equation RMSE = {rmse_ne}")
#print(f"Normal Equation R^2 = {r2_ne}")
#print(f"Normal Equation MAE = {mae_ne}")

plt.figure()
plt.scatter(y_test, y_pred_normal, alpha=0.5)
plt.xlabel("Actual Negative Ratings")
plt.ylabel("Predicted Negative Ratings (Normal Eq.)")
plt.title("Normal Equation Predictions vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.show()

#-----------------Class-5-----------------#
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# X_train, X_test, y_train, y_test = train_test_split(
#    X_scaled, y, test_size=0.2, random_state=42
#) # was done in section 4, doesn't need to be done again

# kNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
# Evaluate kNN
knn_pred = knn.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, knn_pred))
norm_rmse_knn = rmse_knn / (y_test.max() - y_test.min())
r2_knn = r2_score(y_test, knn_pred)
mae_knn = mean_absolute_error(y_test, knn_pred)
#print(f"kNN Root Mean Squared Error = {rmse_knn}")
print(f"kNN Normalized Root Mean Squared Error = {norm_rmse_knn}")
#print(f"kNN R^2 Score = {r2_knn}")
#print(f"kNN Mean Absolute Error = {mae_knn}")


# Random forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Evaluation
pred = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred))
norm_rmse_rf = rmse_rf / (y_test.max() - y_test.min())
r2_rf = r2_score(y_test, pred)
#print(f"Root Mean Squared Error = {rmse_rf}")
print(f"Random Forest Normalized Root Mean Squared Error = {norm_rmse_rf}")
#print(f"R^2 Score = {r2_rf}")
mae = mean_absolute_error(y_test, pred)
#print(f"Mean Absolute Error = {mae}")

# graphing predected vs actual
import matplotlib.pyplot as plt
plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel("Actual Negative Ratings")
plt.ylabel("Predicted Negative Ratings")
plt.title("Random Forest Predictions vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()

results = pd.DataFrame({
    "Model": ["Linear Regression", "Normal Equation", "kNN", "Random Forest"],
    "RMSE": [rmse_lr, rmse_ne, rmse_knn, rmse_rf],
    "MAE": [mae_lr, mae_ne, mae_knn, mae],
    "R2": [r2_lr, r2_ne, r2_knn, r2_rf]
})
results = results.sort_values(by="RMSE")
print("\nModel Comparison Table:")
print(results)
