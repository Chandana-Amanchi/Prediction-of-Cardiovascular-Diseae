# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-rf-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        # Get form values
        age = request.form.get('age')
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = request.form.get('trestbps')
        chol = request.form.get('chol')
        fbs = request.form.get('fbs')
        restecg = request.form.get('restecg')
        thalach = request.form.get('thalach')
        exang = request.form.get('exang')
        oldpeak = request.form.get('oldpeak')
        slope = request.form.get('slope')
        ca = request.form.get('ca')
        thal = request.form.get('thal')

        # Check for missing values
        try:
            # Convert form values to expected data types
            data = np.array([[int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), 
                              int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), 
                              int(ca), int(thal)]])
            my_prediction = model.predict(data)
            return render_template('result.html', prediction=my_prediction)
        except ValueError:
            return render_template('error.html', message="Please fill out all fields.")

        
        

if __name__ == '__main__':
	app.run(debug=True)

