import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df1 = pd.read_csv('train.csv')
gender = {'male': 1,'female': 0}
df1['Sex'] = [gender[item] for item in df1['Sex']]
X = df1[['Fare','Pclass','Parch','SibSp','Sex']]
y = df1['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6)
# df2 = pd.read_csv('test.csv')
# df2['Sex'] = [gender[item] for item in df2['Sex']]
# X_test_output = df2[['Fare','Sex','Pclass','Parch']]
# y_test_output = df2['PassengerId']
plt.scatter(df1['Age'],y)
plt.scatter(df1['Parch'],y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
print("Model accuracy: ",model.score(X_test,y_test))
plt.show()
