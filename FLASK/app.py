import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)
model = load_model("vitamin_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']

        basepath=os.path.dirname(__file__)
        print(f"current path : {basepath}")
        filepath = os.path.join(basepath,'uploads',f.filename)
        print(f"upload folder is : {filepath}")
        f.save(filepath)

        img = image.load_img(filepath,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        preds=model.predict(x)

        labels = ["Vitamin A","Vitamin B","Vitamin C","Vitamin D","Vitamin E"]

        index=np.argmax(preds)

        text = f"The Food Contains : {labels[index]}"
    return text

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
    
if __name__=='__main__':
    app.run(debug=False,threaded=False)
        



