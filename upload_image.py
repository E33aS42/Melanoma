#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from pathlib import PosixPath
import fastai
from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image

# In command line, type:
# $ FLASK_APP=upload_image.py flask run


print(os.getcwd())
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
path = PosixPath('/media/e33as/3A81-FD60/Hackathon/melanoma/DermMel_API')
learn = load_learner(path, 'mel_res34_size224.pkl')
PHOTOS = os.path.join('static', 'photos')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PHOTOS'] = PHOTOS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
          
#@app.route('/', methods=['GET', 'POST'])
#def index():
#    return render_template('index.html')            

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            pathnb = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pathnb)
            
            img = open_image(pathnb)
            pred_class,pred_idx,outputs = learn.predict(img)
            print('class: %s, %s, %s' %(pred_class, pred_idx.item(), outputs[1]))
            return render_template("result.html", category=pred_class, proba=round(outputs[pred_idx.item()].item()*100,2), image=pathnb)
    return render_template("upload.html", logo=os.path.join(app.config['PHOTOS'], 'Logo.jpg'))

if __name__ == "__main__":
    app.run(debug=True)
