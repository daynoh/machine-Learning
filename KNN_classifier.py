import numpy as np
from sklearn import cross_validation,preprocessing,neighbors,svm
import pandas as pd
import matplotlib.pyplot as plt
#preprocessing is transforming raw data to understandable format by comp

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace ('?',-99999, inplace = True)
df.drop(['id'],1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)


print(accuracy)

# making a predictor

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,4,2,3,2,1]])
example_measures = example_measures.reshape(2,-1)
#reshaping is simply letting numpy predict how the matrix should look with -1 specifying the unknown side
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)