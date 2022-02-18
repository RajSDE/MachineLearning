import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cars = pd.read_csv('cars.csv')
x = cars['Year']
y = cars['Price']

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

print('X')
print(x)
print('Y')
print(y)

from sklearn import linear_model
regr = linear_model.LinearRegression()
model = regr.fit(x,y)

# slope: m, intercept: c
m = model.coef_
c = model.intercept_

y_predict = model.predict(x)
print(y_predict)
plt.scatter(x,y)
plt.plot(x,y_predict)
plt.show()
print('x=2022')
print('y_predict =', model.predict([[2022]]))