
import tensorflow
#import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, GlobalAvgPool2D, Flatten, Rescaling #THIS FOR SOME REASON WORKS BETTER
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_data_path, val_data_path):
    preprocessor = ImageDataGenerator(
        rescale = 1/ 255.
    )
    train_generator = preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(512,512),
        color_mode="rgb",
        shuffle=True,
        batch_size=5
    )
    val_generator = preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(512,512),
        color_mode="rgb",
        shuffle=False,
        batch_size=5
    )
    return train_generator, val_generator


def mushroom_model():
    my_input = Input(shape=(512,512,3)) #Images are 512x512 RGB

   # x = Rescaling(1./255)(my_input)
    x = Conv2D(32,(3,3), activation = 'relu')(my_input)
    #x = Conv2D(64,(3,3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,(3,3), activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    #x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(215, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model 

def split_data(split_size, path_to_save_train, path_to_save_val):
    directory = os.listdir(file_path) # list of all mushroom names
    for file in directory:

        path_to_folder = os.path.join(file_path, file)
        image_paths = glob.glob(os.path.join(path_to_folder, '*.png'))
        #Problem: need to split x_train so we have test data as well as validatation data
        x_train, x_val = train_test_split(image_paths, test_size=0.2) # will be integer
        #x_train, x_val = train_test_split(x_train, test_size=split_size)

        #THIS CODE SAVES PICS TO A SPECIFIC FOLDERS
        for x in x_train:

            path_for_train_pic = os.path.join(path_to_save_train, file) # create folder name where train data goes for each mushroom

            if not os.path.isdir(path_for_train_pic):
                os.makedirs(path_for_train_pic)

            shutil.copy(x, path_for_train_pic)#copy image from original folder into train folder

        for x in x_val:

            path_for_train_pic = os.path.join(path_to_save_val, file) # create folder name where train data goes for each mushroom

            if not os.path.isdir(path_for_train_pic):
                os.makedirs(path_for_train_pic)

            shutil.copy(x, path_for_train_pic)#copy image from original folder into train folder            

if __name__ =="__main__":
    file_path = 'D:\\Machine Learning\\kaggle\\mushroom data set\\data\\data'
    path_to_save_train = 'D:\\Machine Learning\\kaggle\\mushroom data set\\data\\train'
    path_to_save_val = 'D:\\Machine Learning\\kaggle\\mushroom data set\\data\\val'

    train_generator, val_generator = create_generators(path_to_save_train,path_to_save_val)


    #split_data(0.1, path_to_save_train, path_to_save_val) # COMMENTED CAUSE WE NO LONGER NEED
    #x_train_array = np.array([])
    #x_train_list = []
    #x_test_list = []

    """""
    for image in x_train:
        image = tensorflow.io.read_file(image)
        image = tensorflow.io.decode_png(image)
        image = tensorflow.image.resize(image, [512,512], method = 'nearest')
        x_train_list.append(image)
        #x_train_array = np.append(x_train_array, image)
    
    for image in x_test:
        image = tensorflow.io.read_file(image)
        image = tensorflow.io.decode_png(image)
        image = tensorflow.image.resize(image, [512,512], method = 'nearest')
        x_test_list.append(image)
    """
    
    model = mushroom_model()
    #model.summary()
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, batch_size=20, epochs=3, validation_data=val_generator)
    #model.evaluate(x_test_list, batch_size=64)
    

#for folder in directory:

