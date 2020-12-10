# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:28:48 2020

@author: KANNAN
"""

from flask import Flask,render_template,request
from keras.models import load_model
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename


model = load_model("Malaria_best_model.hdf5")


app = Flask(__name__)
@app.route('/')

def home():
    return render_template('Malaria.html')


@app.route("/Predict",methods=['GET', 'POST'])
def Malaria():  
    if request.method == "POST": 
        f = request.files['inpFile']
        f.save(secure_filename(f.filename))
        img = load_img(f.filename, target_size=(224,224))
        # preprocess image
        x = img_to_array(img)
        # scaling
        x = x / 255
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        prediction = model.predict(x)
        preds = np.argmax(prediction, axis = 1)
        if preds[0] == 0:
            text  = "You are infected with Malaria"
        else:
            text = "You are uninfected"
        return render_template('Malaria.html',output = text)
    return render_template('Malaria.html')

if __name__ == '__main__':
    app.run()