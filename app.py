# app.py (Flask application)

from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
#from sklearn.externals import joblib
import joblib



app = Flask(__name__)


model = load_model('aayush_xray_classif.h5')

label_encoder = joblib.load('label_encoder.joblib')


'''
from tensorflow.python.keras.models import model_from_json


with open('../Notebooks/model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)


loaded_model.load_weights('../Notebooks/model_weights.h5')


loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        file = request.files['file']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions[0])

        predicted_disease = label_encoder.inverse_transform([predicted_label])[0]


        
        return render_template('result.html', predicted_disease=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)
