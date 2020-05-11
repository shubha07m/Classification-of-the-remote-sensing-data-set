import os
import cv2
import random
from keras.preprocessing import image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.decomposition import PCA
from os import getcwd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

Datadir = os.path.join(getcwd() + '/' + 'dataset' '/' + 'train')
categories = os.listdir(Datadir)
vgg16_feature_list = []
label_list = []
model_vgg16 = VGG16(weights='imagenet', include_top=False)
for category in categories:
    path = os.path.join(Datadir + '/' + category)  # path to the categories
    class_num = category
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path + '/' + img)
            img = image.load_img(img_path, target_size=(256, 256))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            vgg16_feature = model_vgg16.predict(img_data)
            vgg16_feature_np = np.array(vgg16_feature)
            vgg16_feature_list.append(vgg16_feature_np.flatten())
            label_list.append(class_num)

        except Exception as e:
            pass

vgg16_feature_list_np = np.array(vgg16_feature_list)

#### Applying PCA on extracted features using VGA ###

pca = PCA(n_components=15)
#
pcadata = pca.fit_transform(vgg16_feature_list_np)
#

# print(pca.explained_variance_ratio_)

le = LabelEncoder()
target = le.fit_transform(label_list)
target = (np.reshape(target, (len(target), 1)))
X_train, X_test, Y_train, Y_test = train_test_split(pcadata, target, test_size=0.2, random_state=42, shuffle=True)

### applying SVM on PCA data ###

from sklearn.svm import SVC

svm = SVC()
target_pred = svm.fit(X_train, Y_train).predict(X_test)
accuracy = accuracy_score(Y_test, target_pred)
print("accuracy using SVM after applying PCA on VGG feature extracted data is: ")
print(str(accuracy * 100)[:5] + '%')

### applying Naive Bays on PCA data ###
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
target_pred = gnb.fit(X_train, Y_train).predict(X_test)
accuracy = accuracy_score(Y_test, target_pred)
print("accuracy using Naive Bays after applying PCA on VGG feature extracted data is: ")
print(str(accuracy * 100)[:5] + '%')

### applying Random Forest on PCA data ###

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
target_pred = rfc.fit(X_train, Y_train).predict(X_test)
accuracy = accuracy_score(Y_test, target_pred)
print("accuracy using Random Forest after applying PCA on VGG feature extracted data is: ")
print(str(accuracy * 100)[:5] + '%')
