import tensorflow as tf
import keras
import numpy as np
from mnist import MNIST
from sklearn import linear_model
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

print(tf.__version__)
# Read the data from the student-mat csv file and put seperator = ;
data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())
# Slicing the dataset. We have 33 total attributes. Out of it we are choosing 8 attributes for our example
data = data[["G1", "G2", "G3", "absences", "age", "studytime", "failures"]]
# Printing the sliced or picked column value dataset
# print(data.head())
# Let's try to predict the G3 value which is now a label, which we try to predict
# Now we setup two variables. X being a label(G3) in our case and remaining variables are the attributes.
#
predict = "G3"
# X is the features or attributes. In that drop the G3 column values and put all the remaining values. Training data
X = np.array(data.drop([predict], 1))
# Y is the label.
Y = np.array(data[predict])
# splitting the X,Y dataset into four. x_train, y_train, x_test, y_test. test_size = 0.1 mean splitting take 10% of data
# and put it in test.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
# Linear regression is used when the data directly co relate to. y = mx+c
best_score = 0
"""
# Run the model for 50 iterations until the best score or best accuracy has been found. 
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()
    # Fitting the line data into the x and y train values
    linear.fit(x_train, y_train)
    # Finding the accuracy of the model given the test values
    accuracy = linear.score(x_test, y_test)
    if accuracy > best_score:
        print(accuracy)
        best_score = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
# We are getting different accuracy rate because we are picking different test and train data at each iterations.
# This is how we load the saved studentmodel using pickle.
print("best:", best_score)"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

accuracy = linear.score(x_test, y_test)

print(accuracy)
# now we can print all the coefficients of the variables. 6 attributes need 6 coefficients to represent a line in
# multidimensional space.
print("Coeff:", linear.coef_)
# all the intercept
print("Intercept:",linear.intercept_)
# Now we use it actually predict values

prediction = linear.predict(x_test)

for i in range(len(prediction)):
    # y_test[i] is the actual value, x_test is the data for which it has predicted prediction[i]
    print(prediction[i], x_test[i], y_test[i])


style.use("ggplot")
plt.figure()
# Change the p values to different attributes and try to interpret the co relations between the variables.
# If we have plotted study time vs final grade G3, we can interpret that how the study time affect the final grade.
p = "G2"
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final_grade:G3")
plt.show()
