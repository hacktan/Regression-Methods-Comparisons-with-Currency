# -*- coding: utf-8 -*-
"""


@author: hakta
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_dataset.csv",sep = ";")
df=df[::-1]
df['dolar']=pd.to_numeric(df['dolar'].str.replace(',', '.'), errors='coerce')
df['euro']=pd.to_numeric(df['euro'].str.replace(',', '.'), errors='coerce')

x = df.iloc[:,[0,2]].values
y = df.dolar.values.reshape(-1,1)



# %% fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)

# predict
multiple_linear_regression.predict(np.array([[114,4]]))

