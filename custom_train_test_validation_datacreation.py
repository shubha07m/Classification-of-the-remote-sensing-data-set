import os
from os import getcwd
import shutil

Datadir = getcwd()
Source_dir = Datadir + "\\NWPU-RESISC45\\"
Dest_dir = Datadir + "\\newdata\\"
categories = os.listdir(Source_dir)


def divide_datasets(cat_limit, train_size, test_size, validation_size):
    for category in categories[:cat_limit]:
        source_path = os.path.join(Source_dir + category)
        dest_path = os.path.join(Dest_dir + category)
        dest_path_train = os.path.join(Dest_dir + category + '\\train')
        dest_path_test = os.path.join(Dest_dir + category + '\\test')
        dest_path_validation = os.path.join(Dest_dir + category + '\\validation')
        os.mkdir(dest_path)
        os.mkdir(dest_path_train)
        os.mkdir(dest_path_test)
        os.mkdir(dest_path_validation)

        for img in os.listdir(source_path)[:train_size]:
            full_file_name = os.path.join(source_path, img)
            try:
                shutil.copy(full_file_name, dest_path_train)
            except Exception as e:
                pass
        for img in os.listdir(source_path)[train_size:train_size + test_size]:
            full_file_name = os.path.join(source_path, img)
            try:
                shutil.copy(full_file_name, dest_path_test)
            except Exception as e:
                pass
        for img in os.listdir(source_path)[train_size + test_size:train_size + test_size + validation_size]:
            full_file_name = os.path.join(source_path, img)
            try:
                shutil.copy(full_file_name, dest_path_validation)
            except Exception as e:
                pass


used_categories = int(input("How many categories you want to use?\n"))
training_size = int(input("What is the size of training data you want to use?\n"))
testing_size = int(input("What is the size of test data you want to use?\n"))
validationdata_size = int(input("What is the size of validation data you want to use?\n"))
divide_datasets(used_categories, training_size, testing_size, validationdata_size)