#%% Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

#%% Load the Auto MPG dataset
data = sm.datasets.get_rdataset("mtcars").data

# Prepare the data
X = data.drop(columns=['mpg'])
y = data['mpg']

# Add a constant to the model (intercept)
X = sm.add_constant(X)

#%% Fit the linear model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

#%% Get the residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Standardized residuals
influence = model.get_influence()
standardized_residuals = influence.resid_studentized_internal

# Leverage values
leverage = influence.hat_matrix_diag

#%% 1. Linearity Assumption
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    if i < len(X.columns) - 1:  # Exclude the constant term
        sns.scatterplot(x=X.iloc[:, i + 1], y=standardized_residuals, ax=ax)
        ax.set_xlabel(X.columns[i + 1])  # Skip constant column
        ax.set_ylabel('Standardized Residuals')
        ax.axhline(0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

#%% 2. Constant Variance Assumption
plt.scatter(fitted_values, standardized_residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Standardized Residuals')
plt.title('Homoscedasticity Check')
plt.show()

# 3%% Independence Assumption (Durbin-Watson Test)
dw = sm.stats.durbin_watson(standardized_residuals)
print(f'Durbin-Watson statistic: {dw}')

#%% Plot residuals over time to visually inspect independence
plt.plot(standardized_residuals)
plt.xlabel('Index')
plt.ylabel('Standardized Residuals')
plt.title('Standardized Residuals vs. Index')
plt.show()

#%% 4. Normality Assumption
# Histogram of standardized residuals
plt.hist(standardized_residuals, bins=30, edgecolor='k')
plt.xlabel('Standardized Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Standardized Residuals')
plt.show()

# Q-Q plot
fig = plt.figure(figsize=(6, 6))
stats.probplot(standardized_residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Shapiro-Wilk test for normality
stat, p = stats.shapiro(standardized_residuals)
print(f'Statistic: {stat}, p-value: {p}')
