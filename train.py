from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.python import keras
from tensorflow.python.keras import Input, regularizers
from tensorflow.python.keras.backend import shape
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout

currentDir = "D:/Pyhton Projects/Python Gesture Recognition"

classifier = Sequential()

# classifier.add(Input(shape = (64, 64, 1)))
# #classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))
# classifier.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=regularizers.l2(l=0.01)))
# classifier.add(MaxPooling2D(pool_size = (2,2)))

# classifier.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer=regularizers.l2(l=0.01)))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# classifier.add(Flatten())

# classifier.add(Dense(units = 128, activation = 'relu', kernel_regularizer=regularizers.l2(l=0.01)))
# classifier.add(Dense(units = 5, activation = 'softmax'))

# earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=0, mode='auto')

classifier.add(Input(shape = (64, 64, 1)))
classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
classifier.add(Dropout(0.3))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
classifier.add(Dropout(0.3))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 8, activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 5, activation = 'softmax'))

earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=0, mode='auto')

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



trainDataGenerated = ImageDataGenerator(
                                        rescale = 1. / 255, 
                                        shear_range = 0.2, 
                                        zoom_range = 0.2, 
                                        horizontal_flip = True)
testDataGenerated = ImageDataGenerator(rescale = 1. / 255)

trainingSet = trainDataGenerated.flow_from_directory(
                                                    currentDir + '/data/train', 
                                                    target_size = (64, 64), 
                                                    batch_size = 5, 
                                                    color_mode = 'grayscale', 
                                                    class_mode = 'categorical')

testSet = testDataGenerated.flow_from_directory(
                                                currentDir + '/data/test',
                                                target_size = (64, 64), 
                                                batch_size = 5, 
                                                color_mode = 'grayscale', 
                                                class_mode = 'categorical')

# classifier.fit_generator(
#                         trainingSet,
#                         steps_per_epoch = 600,
#                         epochs = 10,
#                         validation_data = testSet,
#                         validation_steps = 30)

import matplotlib.pyplot as plt

def plot_history(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc ="lower right")
    axs[0].set_title("Accuracy Eval")


    axs[1].plot(history.history["loss"], label = "train error")
    axs[1].plot(history.history["val_loss"], label = "test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc ="upper right")
    axs[1].set_title("Error Eval")

    plt.show()

history = classifier.fit_generator(trainingSet, validation_data = testSet, epochs = 50, callbacks=[earlystop])

plot_history(history)

roundedPredictions = classifier.predict_classes(testSet, batch_size = 5, verbose = 0)
print("Length " + str(len(roundedPredictions)))

from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

cm = confusion_matrix(testSet.labels, roundedPredictions)

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')   
    plt.show()

cm_plot_labels = ['Peace', 'Palm', 'Fist', 'Thumbs Up', 'Ok']     
plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion Matrix')

# classifier.fit(trainingSet,
#                validation_data = testSet,
#                epochs = 10)

model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')