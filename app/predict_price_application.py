import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# yahoo finance used to fetch data
import yfinance as yf

yf.pdr_override()

options = " AAPL Stock Linear Regression Prediction, AAPL Stock Logistic Regression Prediction, AAPL Desicion Tree Regression, AAPL Bayesian Ridge Regression, Exit".split(
    ",")


# Input Start Date
def start_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    start = datetime.datetime.strptime(date_entry, '%m/%d/%Y')
    start = start.strftime('%Y-%m-%d')
    return start


# Input End Date
def end_date():
    date_entry = input('Enter a ending date in MM/DD/YYYY format: ')
    end = datetime.datetime.strptime(date_entry, '%m/%d/%Y')
    end = end.strftime('%Y-%m-%d')
    return end


# Input Symbols
def input_symbol():
    symbol = input("Enter symbol: ").upper()
    return symbol


# Logistic Regression
def stock_logistic_regression():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)

    X = df.loc[:, df.columns != 'Adj Close']
    y = np.where(df['Adj Close'].shift(-1) > df['Adj Close'], 1, -1)

    split = int(0.7 * len(df))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = LogisticRegression()
    model = model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))
    print(model.score(X_test, y_test))
    cross_val = cross_validate(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print('_____________Summary:_____________')
    print('Estimate intercept coefficient:', model.intercept_)
    print('Number of coefficients:', len(model.coef_))
    print('Accuracy Score:', cross_val['test_score'].mean())
    print("")
    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()

    # Linear Regression


def stock_linear_regression():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    n = len(df.index)
    X = np.array(df['Open']).reshape(n, -1)
    Y = np.array(df['Adj Close']).reshape(n, -1)
    lr = LinearRegression()
    lr.fit(X, Y)
    lr.predict(X)

    plt.figure(figsize=(12, 8))
    plt.scatter(df['Adj Close'], lr.predict(X))
    plt.plot(X, lr.predict(X), color='red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices')
    plt.show()
    print('_____________Summary:_____________')
    print('Estimate intercept coefficient:', lr.intercept_)
    print('Number of coefficients:', len(lr.coef_))
    print('Accuracy Score:', lr.score(X, Y))
    print("")
    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()

    # Support Vector Regression


def decision_tree():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)

    # dates = np.reshape(df.index, (len(df.index), 1))  # convert to 1xn dimension
    n = len(df.index)
    dates = np.array(df['Adj Close']).reshape(n, -1)
    x = np.array(df['Open']).reshape(n, -1)
    prices = df['Adj Close']

    # Initialize decision tree regressor
    dt_regressor = DecisionTreeRegressor()

    # Fit regression model
    dt_regressor.fit(dates, prices)

    plt.figure(figsize=(12, 8))
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, dt_regressor.predict(dates), c='g', label='Decision Tree model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.show()

    predictions = dt_regressor.predict(dates)
    rmse = np.sqrt(mean_squared_error(prices, predictions))
    r2 = r2_score(prices, predictions)

    print('_____________Summary:_____________')
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Coefficient of Determination (R-squared):', r2)
    print('Accuracy Score:', dt_regressor.score(x, prices))
    print("")
    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()

        # Support Vector Regression

        # Bayesian Ridge Regression


def bayesian_ridge_regression():
    s = start_date()
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    n = len(df.index)
    x = np.array(df['Open']).reshape(n, -1)
    y = np.array(df['Adj Close']).reshape(n, -1)
    br = BayesianRidge()
    br.fit(x, y)
    br.predict(x)

    plt.figure(figsize=(12, 8))
    plt.scatter(df['Adj Close'], br.predict(x))
    plt.plot(x, br.predict(x), color='red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices')
    plt.show()
    print('_____________Summary:_____________')
    print('Estimate intercept coefficient:', br.intercept_)
    print('Number of coefficients:', len(br.coef_))
    print('Accuracy Score:', br.score(x, y))
    print("")
    ans = ['1', '2']
    user_input = input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()



# ******************************************************* Menu **********************************************************#
# ***********************************************************************************************************************#
def menu():
    ans = ['1', '2', '3', '4', '5', '0']
    print(""" 
              
                           MENU
            MACHINE LEARNING STOCK PRICE PREDICTION        
                  ---------------------------
                  1.Linear Regression
                  2.Logistic Regressions
                  3.Decision Tree
                  4.Bayesian Ridge Regression
                  5.Beginning Menu
                  0.Exit the Program
                  """)
    user_input = input("Command (0-4): ")
    while user_input not in ans:
        print("Error: Please enter a valid option 0-3")
        user_input = input("Command: ")
    if user_input == '1':
        stock_linear_regression()
    elif user_input == '2':
        stock_logistic_regression()
    elif user_input == '3':
        decision_tree()
    elif user_input == "4":
        bayesian_ridge_regression()
    elif user_input == "5":
        beginning()
    elif user_input == "0":
        exit()

    # ***********************************************************************************************************************#


# *************************************************** Start of Program **************************************************#
# ***********************************************************************************************************************#
def beginning():
    print()
    print("----------Welcome to Machine Learning AAPL Stock Price Predictions--------")
    print("""
Please choose option 1 or 2
              
1. Menu
2. Exit Program 
---------------------------------------------""")
    ans = ['1', '2']
    user_input = input("What is your Option?: ")
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input = input("Command: ")
    if user_input == "1":
        menu()
    elif user_input == "2":
        exit()


# ***********************************************************************************************************************#
beginning()

# %%

# %%
