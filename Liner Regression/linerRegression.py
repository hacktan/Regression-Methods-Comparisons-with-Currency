# -*- coding: utf-8 -*-
"""


@author: hakta
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("USD_TRY_Gecmis_Verileri.csv",sep = ";")

df['Acilis']=pd.to_numeric(df['Acilis'].str.replace(',', '.'), errors='coerce')
df=df[::-1]

plt.scatter(df.Tarih,df.Acilis)
plt.xlabel("Gun")
plt.ylabel("Kur")
plt.show()

#%% linear regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.Tarih.values.reshape(-1,1)
y = df.Acilis.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% prediction

gunler = df.Tarih.tolist()

import numpy as np

b0 = linear_reg.predict([[0]])
print("b0: ",b0)

b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept

b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope


print(linear_reg.predict([[11]]))

# visualize line
array = np.array(gunler).reshape(-1,1)  

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)  

plt.plot(array, y_head,color = "red")

linear_reg.predict([[100]])

#%% R2 Score

from sklearn.metrics import r2_score

print("r_square score for Linear Regression:",r2_score(y,y_head))






















