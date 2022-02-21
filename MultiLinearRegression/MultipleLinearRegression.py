#  Multiple linear regression on custom design car dataset


import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('cars.csv')

x = np.array(df[['Weight','Volume']])
y = np.array(df['CO2'])

regr = linear_model.LinearRegression()
regr.fit(x,y)
predicted_y = regr.predict(x)

data = {'Car':df['Car'],'Weight':df['Weight'],'Volume':df['Volume'],'CO2':df['CO2'],'Predicted CO2':predicted_y}

pred_df = pd.DataFrame(data)
print('Results \n========\n')
print(pred_df)

wt = 1200
vol = 1300
pred_result = regr.predict([[wt,vol]])
print('For new weight', wt,'KG and New Volume=',vol,'cm3;The CO2 emission will be',pred_result[0],'g(approx)')