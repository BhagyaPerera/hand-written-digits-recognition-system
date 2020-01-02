from sklearn.datasets import load_digits

#digits-a toy dataset available in scikit learn

dataset=load_digits()

data=dataset.data
target=dataset.target
imgs=dataset.images#images have the actual pixel values


#thresholding imgs to get black and white pixels
import cv2

ret,data=cv2.threshold(data,6,15,cv2.THRESH_BINARY)
ret,imgs=cv2.threshold(imgs,6,15,cv2.THRESH_BINARY)




print(imgs[0])#The pixel values are in 0-15.So each pixel contains maximum of 4bits number. Therefore we have to convert it into a 8bits.
print(data.shape)
print(target.shape)
print(imgs.shape)



#train the dataset
##from sklearn.model_selection import train_test_split
##
##
##
##train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

from sklearn.svm import SVC

clsfr=SVC(kernel='linear')



clsfr.fit(data,target)

import joblib

joblib.dump(clsfr,'SVM_DIGITS.sav',)






##results=clsfr.predict(test_data)
##
##
##from sklearn.metrics import accuracy_score
##
##accuracy=accuracy_score(test_target,results)
##print('Accuracy:',accuracy)
##
##
##
##
##
##
##
##from matplotlib import pyplot as plt
##plt.imshow(imgs[0],cmap='gray')
##plt.show()
