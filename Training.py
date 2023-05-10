import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import warnings
import numpy as np
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)




# train_path = r'.\Dataset\train'
# test_path = r'.\Dataset\test'
train_path = r'.\Indian_split\train'
test_path = r'.\Indian_split\test'

def to_grayscale_then_rgb(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

# tf.keras.applications.vgg16.preprocess_input
train_batches = ImageDataGenerator(preprocessing_function=to_grayscale_then_rgb).flow_from_directory(directory=train_path, target_size=(64, 64), class_mode='categorical', batch_size=50, shuffle=True)
test_batches = ImageDataGenerator(preprocessing_function=to_grayscale_then_rgb).flow_from_directory(directory=test_path, target_size=(64, 64), class_mode='categorical', batch_size=50, shuffle=True)
print("here")
# 50 images and their labels
imgs, labels = next(train_batches)


# Plotting the images...

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 2, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img.astype('uint8'))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(imgs)
print(imgs.shape)
print(labels)
# exit(0)



# Applying one model after another in sequential manner
model = Sequential()    

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding= 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding= 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

# 64 neurons in first hidden layer

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation ="relu"))
model.add(Dropout(0.3))
model.add(Dense(35, activation ="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


print("--------")
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop],  validation_data = test_batches, validation_steps=5)#, checkpoint])
imgs, labels = next(train_batches) # For getting next batch of imgs...

imgs, labels = next(test_batches) # For getting next batch of imgs...
scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')


#model.save('best_model_dataflair.h5')
model.save('model_Indian_split.h5')

print(history2.history)

imgs, labels = next(test_batches)

model = keras.models.load_model("./model_Indian_split.h5")

scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

model.summary()
model.metrics_names


word_dict = {0:'One',1:'Two',2:'Three',3:'Four',4:'Five',5:'Six',6:'Seven',7:'Eight',8:'Nine', 9:'A', 10: 'B',
             11:'C',12:'D',13:'E',14:'F',15:'G',16:'H',17:'I',18:'J', 19:'K', 20: 'L',
             21:'M',22:'N',23:'O',24:'P',25:'Q',26:'R',27:'S',28:'T', 29:'U', 30: 'V',
             31:'W',32:'X',33:'Y',34:'Z'}

predictions = model.predict(imgs, verbose=0)
print("predictions on a small set of test data--")
print("")
for ind, i in enumerate(predictions):
    print(word_dict[np.argmax(i)], end='   ')

plotImages(imgs)
print('Actual labels')
for i in labels:
    print(word_dict[np.argmax(i)], end='   ')

print(imgs.shape)

print(history2.history)

print("Akshaya")

# Epoch 684