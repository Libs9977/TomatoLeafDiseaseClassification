import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings

from tensorflow.keras.optimizers import SGD

warnings.simplefilter(action='ignore', category=FutureWarning)


train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'
bs = 128


# Classes of the dataset
classes = ['Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Target_spot', 'Spider_mites', 'Septoria_leaf_spot', 'Leaf_mold',
           'Late_blight', 'Healthy', 'Early_blight', 'Bacterial_spot']

datagen = ImageDataGenerator(
    #preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale=1.0 / 255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_batches = datagen.flow_from_directory(directory=train_path, target_size=(224, 224), classes=classes, batch_size=bs)
valid_batches = datagen.flow_from_directory(directory='data/valid',
                                            target_size=(224, 224),
                                            classes=['Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Target_spot', 'Spider_mites', 'Septoria_leaf_spot', 'Leaf_mold', 'Late_blight', 'Healthy', 'Early_blight', 'Bacterial_spot'],
                                            batch_size=bs,
                                            )
test_batches = datagen.flow_from_directory(directory=test_path,
                                           target_size=(224, 224),
                                           classes=classes,
                                           batch_size=bs,
                                           shuffle=False
                                           )
images, labels = next(train_batches)

premodel = VGG16()
model = Sequential()
for layer in premodel.layers[:-1]:
    model.add(layer)
for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=10, activation='softmax'))
model.summary()
#model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x=train_batches, validation_data=valid_batches, epochs=150, verbose=2)

# Compile the model
optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch = valid_batches.samples // valid_batches.batch_size
model.fit(train_batches, validation_data=valid_batches, epochs=5, verbose=2)


images_test, labels_test = next(test_batches)

loss, accuracy = model.evaluate(test_batches)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
model.save('VGG16ModelNoProcessDataLowParameters.keras')


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
    plt.xticks(tick_marks, classes, rotation=45)
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
