import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

df = pd.read_csv('dataset.csv',encoding='latin-1')
y = df['Patientdoesntappeared']
x.drop(['Name','MobileNumber','Patientdoesntappeared'], axis=1, inplace=True)

x_train, x_train, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=17)

svc =LinearSVC()
svc.fit(x_train, y_train)


pickle.dump(svc,open('model.pkl','wb'))