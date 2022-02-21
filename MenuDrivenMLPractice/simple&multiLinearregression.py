# A. Write a menu-driven program in python to perform.
#   1. Simple Linear regression.
#   2. Multiple Linear regression

# defining functions  
def read_dataset():
    cars = pd.read_csv('cars1.csv')
    return cars 
def display_dataset():
    print('\nFetching Dataset', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
    print('X')
    print(x)
    print('Y')
    print(y)   
def simple_linear():
    print('\nApplying Simple Linear Regression', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
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
def multiple_regression():
    print('\nApplying Multiple Linear Regression', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
    import pandas as pd
    import numpy as np
    from sklearn import linear_model

    df = pd.read_csv('cars_dataset1.csv')

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
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
cars = read_dataset()
x = cars['Year']
y = cars['Price']
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

  
print("Simple Liner Regression & Multiple Liner Regression Model")  

# creating options  
while True:  
    print("\nMAIN MENU")  
    print("1. Read the Dataset")  
    print("2. Display the Dataset")
    print("3. Simple Liner Regression")
    print("4. Multiple Linear Regression ") 
    print("5. Exit")  
    choice1 = int(input("Enter the Choice:"))  
  
    if choice1 == 1:  
        print('\nReading the Dataset', end="")
        time.sleep(1)
        print('.', end="")
        time.sleep(1)
        print(' .',end="")
        time.sleep(1)
        print(' .')
        read_dataset()
        time.sleep(1)
        print('Dataset has been loaded')
    elif choice1 == 2:
        read_dataset()  
        display_dataset()
    elif choice1 == 3:
        simple_linear()
    elif choice1 == 4:
        multiple_regression()
    elif choice1 == 5:
        break      
    else:  
        print("Oops! Incorrect Choice.")  