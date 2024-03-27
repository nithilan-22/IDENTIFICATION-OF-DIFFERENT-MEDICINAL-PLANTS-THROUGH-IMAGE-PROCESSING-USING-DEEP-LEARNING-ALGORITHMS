from flask import * 
from tensorflow import *
from keras.models import load_model
from keras_preprocessing import image
import sys
import os
import glob
import re
import numpy as np

model=load_model("Models.h5")
directory="Medicinal Leaf Images"
class_names=[]
for item in os.listdir(directory):
    #print(item)
    class_names.append(item)
#print(class_names)

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method=='POST':
        file=request.files['file']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',file.filename)
        file.save(file_path)
        file_path=f"uploads/{file.filename}"
        test_image=image.load_img(file_path,target_size=(128,128))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=model.predict(test_image)
        #print(result)
        for i in range(len(class_names)):
            if result[0][i]==1:
                output=class_names[i]
        return output
    
if __name__=='__main__':
    app.run(debug=True)