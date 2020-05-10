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

Datadir = os.path.join(getcwd() + '/' + 'dataset' + '/' + 'train')
categories = os.listdir(Datadir)
vgg16_feature_list = []

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

        except Exception as e:
             pass

vgg16_feature_list_np = np.array(vgg16_feature_list)

pca = PCA(n_components=2)

pcadata = pca.fit_transform(vgg16_feature_list_np)

print(pcadata)