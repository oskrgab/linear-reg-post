#%% Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import statsmodels.api as sm

#%% Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

#%% Fit linear model
model = LinearRegression()
model.fit(X, y)

#%% Predicted values
y_pred = model.predict(X)
residuals = y - y_pred

#%% Linearity Assumption
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    if i < len(X.columns):
        ax.scatter(X.iloc[:, i], residuals)
        ax.set_xlabel(X.columns[i])
        ax.set_ylabel('Residuals')
        ax.axhline(0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

#%% Constant Variance Assumption
plt.scatter(y_pred, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity Check')
plt.show()

#%% Independence Assumption
dw = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw}')

#%% Normality Assumption
stat, p = shapiro(residuals)
print(f'Statistic: {stat}, p-value: {p}')

#%% Q-Q plot
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot')
plt.show()
#%% Plot histogram
sns.histplot(residuals)
plt.show() 

# %%
