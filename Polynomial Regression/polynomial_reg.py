# -*- coding: utf-8 -*-
"""


@author: hakta
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("2016dolaralis.csv",sep = ",")

x = df.Gun.values.reshape(-1,1)
y = df.Fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Dolar Kuru")
plt.xlabel("Gun")
plt.show()

# linear regression =  y = b0 + b1*x
# multiple linear regression   y = b0 + b1*x1 + b2*x2

# %% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head = lr.predict(x)

plt.plot(x,y_head,color="red",label ="linear")
plt.show()

print("Linear Regression için 10 000 gün sonra dolar kuru: ",lr.predict([[10000]]))

#%% R2 Score with Polynomial

from sklearn.metrics import r2_score

print("r_square score for Linear Regression:",r2_score(y,y_head))
# %%
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)


# %% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

# %%

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()
#print(linear_regression2.predict([[10000]]))

#%% R2 Score with Polynomial

from sklearn.metrics import r2_score

print("r_square score for Polynomial Regression:",r2_score(y,y_head2))


















