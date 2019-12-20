import numpy as np
import matplotlib.pyplot as plt
# seaborn is good for heatmap. Brighter red hotter it is.
import seaborn as sns; sns.set()
import sklearn as sk
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

train = np.loadtxt('train.txt')
test = np.loadtxt('test.txt')

model = MultinomialNB()
model.fit(train[:,0:3], train[:,4] )
labels =  model.predict(test[:,0:3])

mat = confusion_matrix(test[:,4], labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d')
plt.xlabel('true label')
plt.ylabel('predicted table')
plt.show()

# def predict_cat(s,train=train, model=model):
#     pred = model.predict(s)
#     return train.