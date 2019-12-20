import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

data = pd.read_csv('car.data')

print(data.head())

# Data's are string here. We need to convert them to integers for preprocessing.This can be done
# using sklearn preprocessing methods.

convert = preprocessing.LabelEncoder()

buying = convert.fit_transform(list(data["buying"]))
maint = convert.fit_transform(list(data["maint"]))
door = convert.fit_transform(list(data["door"]))
persons = convert.fit_transform(list(data["persons"]))
lug_boot = convert.fit_transform(list(data["lug_boot"]))
safety = convert.fit_transform(list(data["safety"]))
cls = convert.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)
# Split the total data for training and testing. We can change the test_size.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

print(x_train, x_test, y_train, y_test)

model = KNeighborsClassifier(9)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)

predict = model.predict(x_test)

category = ["unacc", "acc", "good", "vgood"]

for x in range(len(predict)):
    print("Prediction:", category[predict[x]], "Data: ", x_test[x], "Actual:", category[y_test[x]])
    # This prints the nearest 9 neigbours and the distance
    n = model.kneighbors([x_test[x]], 9, True)
    print("N:", n)






