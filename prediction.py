import pandas as pd
import pickle
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3,
},index=[0])

data = pd.read_csv('heart.csv')
x = data.drop('target',axis=1)
y = data['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

data = pd.read_csv('heart.csv')
data = data.drop_duplicates()
x = data.drop('target',axis=1)
y = data['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


X=data.drop('target',axis=1)
y=data['target']
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X,y)

p=rf.predict(new_data)
if p[0]==0:
    print("NO Disease")
else:
    print("Disease")
import joblib
joblib.dump(rf,'model_joblib_heart')
model = joblib.load('model_joblib_heart')
model.predict(new_data)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-rf-model.pkl'
pickle.dump(model, open(filename, 'wb'))

