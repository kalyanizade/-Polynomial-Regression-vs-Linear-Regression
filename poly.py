import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Kalyani\vscode_projects\spyder\Polynomial Regression\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#-----------------------------------------

# linear vs polynominal model


# linear model  -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# predicton 

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

# Linear model is not a good model because we predit sal array([330378.78787879]) we want betn 150000 to 200000


#------------------------------------------

# polynomial model  ( bydefeaut degree - 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures() # bydefault degree 2
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



# poly nomial visualization 

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicton 

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# polynominal model is a good model because we predit salary array([189498.10606061]) we want betn 150000 to 200000



