import os
import pickle
import random
from os import getcwd

import cv2
import numpy as np

Datadir = getcwd() + "\\newdata"

categories = os.listdir(Datadir)

limit = 1000000  # can be customized depending on the size of data we want to use


def create_dataset(dataset_type):
    created_data = []
    if dataset_type == 1:
        part = 'train'
        # created_data = training_data
    elif dataset_type == 2:
        part = 'test'
        # created_data = validation_data
    else:
        part = 'validation'
        # created_data = test_data
    for category in categories:
        if len(created_data) < limit:
            path = os.path.join(Datadir + '/' + category + '/' + part)  # path to the categories
            class_num = category
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img))
                    # new_array = cv2.resize(img_array, (256, 256))

                    created_data.append([img_array, class_num])
                except Exception as e:
                    pass
        random.shuffle(created_data)

    print('created ' + part + ' data size is: ' + str(len(created_data)))
    return created_data


def pick_datasave(dataset_type):
    if dataset_type == 1:
        created_data = create_dataset(1)
    elif dataset_type == 2:
        created_data = create_dataset(2)
    else:
        created_data = create_dataset(3)

    X = []
    Y = []

    for features, label in created_data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, 256, 256, 3)

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle", "rb")
    Y = pickle.load(pickle_in)

    return X, Y


X_train, Y_train = pick_datasave(1)
X_test, Y_test = pick_datasave(2)
X_validation, Y_validation = pick_datasave(3)

# making the model
#
from keras.applications import VGG16
import keras
from keras.applications import VGG16

vgg = VGG16(weights=None, classes=15, input_shape=(256, 256, 3))
vgg.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

vgg.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1)
#
# # #
# # # # Final evaluation of the model
# # scores = our_model.evaluate(X_test, Y_test, verbose=0)
# # print("Accuracy: %.2f%%" % (scores[1] * 100))
# # our_model.save('./our_model' + '.h5')
