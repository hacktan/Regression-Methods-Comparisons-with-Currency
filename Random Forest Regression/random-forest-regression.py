# -*- coding: utf-8 -*-
"""
Created on Sat 

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv("random_forest_dataset.csv",sep = ";")
df=df[::-1]
df['dolar']=pd.to_numeric(df['dolar'].str.replace(',', '.'), errors='coerce')

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)

print("15. Günde Dolar Kurunun Tahmini: ",rf.predict([[15]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Piyasa Günü")
plt.ylabel("USDTRY")
plt.show()

