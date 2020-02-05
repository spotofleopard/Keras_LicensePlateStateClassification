import os,numpy as np
import tensorflow as tf
from PIL import Image

total_imgs=0
correct=0
error=0
classes= ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

model=tf.keras.models.load_model('saved_model.h5')
test_folder=os.path.expanduser('~/dataset/LPR/plates_128x64/test')
html_output='<html><body><table border="1"><tr><th>Image</th><th>GT</th><th>Classifier Output</th><th>Confidence</th></tr>'
for subdir, dirs, files in os.walk(test_folder):
    for file_name in files:
        if file_name.endswith(".jpg"):

            path=os.path.join(subdir, file_name)
            print(path)
            pcs=path.split('/')
            state_gt=pcs[-2]

            image=np.asarray(Image.open(path)).astype(float)/255
            result=model.predict(image.reshape(1,64,128,3))[0]
            idx=np.argmax(result)
            state_output=classes[idx]
            confidence=result[idx]
            total_imgs+=1

            if(state_output!=state_gt):
                bgcolor='#ffcccc'
                error+=1
            else:
                bgcolor='#ccffcc'
                correct+=1
            html_output+='<tr bgcolor="{}"><td><img src="file://{}"></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(bgcolor,path,state_gt,state_output,confidence)

html_output+='<table><h3>accuracy:{}/{},{:.2f}%</h3></body></html>'.format(correct,total_imgs,correct*100.0/total_imgs)
with open('/tmp/plate_state_classification_errors.html','w') as f:
    f.write(html_output)
