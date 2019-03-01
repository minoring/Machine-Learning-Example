#%%
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('precision', '3')

#%%
## Load data set. 
boston = load_boston()
# Convert explanatory variable to DataFrame. 
df = DataFrame(boston.data, columns = boston.feature_names)
# Add Objective variable to DataFrame. 
df['MEDV'] = np.array(boston.target)

#%%
# Show first 5 datas. 
df.head()

#%%
# Create model. Ridge Regularization Regression. 
model = linear_model.Ridge()
# Find params.
model.fit(boston.data,boston.target)

#%%
# Print params. 
print(model.coef_)
print(model.intercept_)

#%%
# Holdout cross validation. 
# 75% trainning data, 25% validation data.
X_train, X_test, y_train, y_test = train_test_split(boston.data, 
                                                    boston.target, 
                                                    test_size = 0.25, 
                                                    random_state = 100)

# Calculate parameters by trainning data.
model.fit(X_train, y_train)

# Predict training, test data by model.
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#%%
# Plot residuals.
plt.scatter(y_train_pred, y_train_pred - y_train, 
            c = 'gray', marker = 'o', label = 'Train Data')
plt.scatter(y_test_pred, y_test_pred - y_test, 
            c = 'blue', marker = 's', label = 'Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')

# Plot y = 0 line. 
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'black')
plt.xlim([0, 50])
plt.show()

#%%
# MSE
print('MSE Train : %.3f, Test : %.3f' % (
        mean_squared_error(y_train, y_train_pred), 
        mean_squared_error(y_test, y_test_pred)))

# Coefficient of determination. R^2
print('R^2 Train : %.3f, Test : %.3f' % (
        model.score(X_train, y_train), 
        model.score(X_test, y_test)))