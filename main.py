import re
import base64
from flask import Flask, render_template,request
import cv2
import numpy as np

import joblib

algorithm=joblib.load('SVM_DIGITS.sav')



app = Flask(__name__)

@app.route('/')
def index():
    global draw_digit
    draw_digit=0
    return render_template('drawdigits.html')

@app.route('/predictdigits/', methods=['GET','POST'])
def predict_digits():
    parseImage(request.get_data())
    
    test_img=cv2.imread('output.png')
    test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    test_img=crop(test_img)
    test_img=cv2.resize(test_img,(8,8))#resizing 8*8
    test_img=np.reshape(test_img,(1,64))#reshaing 1*64
    
    test_img=(test_img/255.0)*15.0#scaling down 0-255 to 0-15
    test_img=15.0-test_img #inverting
    result=algorithm.predict(test_img)
    print(test_img.shape)
    

    
    return str(result)

def crop(im):

    ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #following if statement is to ignore the noises and save the images which are of normal size(character)
        #In order to write more general code, than specifying the dimensions as 100,
        # number of characters should be divided by word dimension            
        if(i==1):
            return thresh1[y:y+h,x:x+w]
        i=i+1

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.run(debug=True)
