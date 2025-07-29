import tensorflow as tf
from keras.src.utils import plot_model
from wandb.integration.keras import WandbCallback

import wandb as wandb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD


train_path = 'dataProcess/train'
valid_path = 'dataProcess/valid'
test_path = 'dataProcess/test'
bs = 128
# Classes of the dataset
classes = ['Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Target_spot', 'Spider_mites', 'Septoria_leaf_spot', 'Leaf_mold',
           'Late_blight', 'Healthy', 'Early_blight', 'Bacterial_spot']

datagen = ImageDataGenerator(
    #preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale=1.0 / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True
)

train_batches = datagen.flow_from_directory(directory=train_path,  target_size=(227, 227), batch_size=bs, class_mode='categorical')
valid_batches = datagen.flow_from_directory(directory=valid_path, target_size=(227, 227), batch_size=bs, class_mode='categorical')
test_batches = datagen.flow_from_directory(directory=test_path, target_size=(227, 227), batch_size=bs, class_mode='categorical',shuffle=False)

model = Sequential([
    Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid'),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(units=10, activation='softmax')
])
model.summary()

# Compile the model
optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_batches, epochs=30, verbose=2)

images_test, labels_test = next(test_batches)

loss, accuracy = model.evaluate(test_batches)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
model.save('model_AlexNet.h5')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=40)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


predictions = model.predict(x=test_batches, verbose=0)
cfm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Target_spot', 'Spider_mites', 'Septoria_leaf_spot',
                  'Leaf_mold', 'Late_blight', 'Healthy', 'Early_blight', 'Bacterial_spot']
plt.figure()
plot_confusion_matrix(cfm, classes=cm_plot_labels, title="Confusion matrix")
plt.show()
