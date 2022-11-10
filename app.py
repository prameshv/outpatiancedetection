from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/predict',methods=['POST'])
def home():
    ear = request.form['ear']
    nose = request.form['nose']
    tounge = request.form['tounge']
    arr = np.array([[ear,nose,tounge]])
    
    input_reshape = arr.reshape(1,-1)
    predict = model.predict(input_reshape)
    
    pred = model.predict(arr)
    
    return render_template('results.html',data=pred)

if __name__ == '__main__':
    app.run(debug=True)
