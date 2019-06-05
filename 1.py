# Simple Linear Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import  LinearRegression
# Create an object of class LinearRegression Class
regressor = LinearRegression()
# Fit the trainging dataset to the regressor
regressor.fit(xTrain, yTrain)


# Predicting the Test set Result
yPred = regressor.predict(xTest)

# Visualising the Training set results using matplotlib
# Plotting the graph of years of experience vs Salary
plt.scatter(xTrain, yTrain, color = 'red')
# Plotting the blue line for the regression line
plt.plot(xTrain, regressor.predict(xTrain), color = 'blue')
# Title of the graph
plt.title('Salary vs Experience {Training set}')
# Labels for the graph
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results using matplotlib
# Plotting the graph of years of experience vs Salary
plt.scatter(xTest, yTest, color = 'red')
# Plotting the blue line for the regression line
plt.plot(xTrain, regressor.predict(xTrain), color = 'blue')
# Title of the graph
plt.title('Salary vs Experience {Test set}')
# Labels for the graph
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()