import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2
import gc
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

from zipfile import ZipFile

data_path = 'data-img.zip'
extracted_path = 'lung_colon_image_set'

if not os.path.exists(extracted_path):
    with ZipFile(data_path, 'r') as zip:
        zip.extractall(extracted_path)
        print('The data set has been extracted.')
else:
    print('The data set is already extracted.')

path = 'lung_colon_image_set/lung_image_sets'
classes = os.listdir(path)
print(classes)

path = 'lung_colon_image_set/lung_image_sets'

for cat in classes:
	image_dir = os.path.join(path, cat)
	images = os.listdir(image_dir)

	fig, ax = plt.subplots(1, 3, figsize=(15, 5))
	fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)

	for i in range(3):
		k = np.random.randint(0, len(images))
		img = np.array(Image.open(f'{path}/{cat}/{images[k]}'))
		ax[i].imshow(img)
		ax[i].axis('off')
	plt.show()

IMG_SIZE = 256
EPOCHS = 15
BATCH_SIZE = 64
SPLIT = 0.30

# initial_learning_rate = 0.01
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=1000,
#     decay_rate=0.65)

X = []
Y = []

for i, cat in enumerate(classes):
    print("Finding Images")
    images = glob(f'{path}/{cat}/*.jpeg')

    for image in images:
        img = cv2.imread(image)
        
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values

X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y,
												test_size = SPLIT,
												random_state = 2022)
print(X_train.shape, X_val.shape)
all_metrics = []


def cnn_model():
    model = keras.models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same', kernel_regularizer=l2(0)),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0)),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0)),
        layers.MaxPooling2D(2, 2),  
        layers.BatchNormalization(),      
        layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0)),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(3, activation='softmax')
    ])
    return model

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=45, 
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.2, 
    zoom_range=0.2,  
    horizontal_flip=True,  
    vertical_flip=True,  
    fill_mode='nearest'  
)

val_datagen = ImageDataGenerator(rescale=1/255)

model = cnn_model()
model.summary()
model.compile(
	optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
)


class CallbackEarly(tf.keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.99:
             print("Sufficiently trained")
             self.model.stop_training = True

es = EarlyStopping(patience=200, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=200, factor=0.99, verbose=1)

best_model = '3_best_trained.h5'
checkpoint = ModelCheckpoint(best_model, monitor = 'val_accuracy', save_best_only = True, mode = 'max', verbose = 1)

history = model.fit(
    train_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    validation_data=val_datagen.flow(X_val, Y_val, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=[es, lr, CallbackEarly(), checkpoint],
    verbose=1
)

metrics_dict = {}
metrics_dict['accuracy'] = history.history['accuracy'][-1]
metrics_dict['val_accuracy'] = history.history['val_accuracy'][-1]
metrics_dict['loss'] = history.history['loss'][-1]
metrics_dict['val_loss'] = history.history['val_loss'][-1]

all_metrics.append(metrics_dict)

avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0]}
print("Average Metrics:")
print(avg_metrics)

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

Y_pred = model.predict(X_val)
Y_val = np.argmax(Y_val, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)

metrics.confusion_matrix(Y_val, Y_pred)

print(metrics.classification_report(Y_val, Y_pred, target_names=classes))