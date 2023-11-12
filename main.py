import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import matplotlib as mpl
import PIL
import PIL.Image

'''
n01443537 goldfish
n01629819 salamander
n01882714 koala
n01910747 jellyfish
n04146614 school bus
n04487081 trolley
n07873807 pizza
n07920052 expresso
'''

#%%

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + '\\dataset', #\\n01629819',
    validation_split=0.2,
    subset="training",
    seed = 9876,
    image_size=(64, 64),
    batch_size=50)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.getcwd() + '\\dataset', #\\n01629819',
    validation_split=0.2,
    subset="validation",
    image_size=(64, 64),
    seed = 9876,
    batch_size= 50)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_images = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_images = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#%%

model_CNN = keras.Sequential([
    keras.layers.Rescaling(1/255, input_shape = (64, 64, 3)),
    keras.layers.Conv2D(258, 3, activation = 'relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(258, 3, activation = 'relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(258, 3, activation = 'relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(258, activation='relu'),
    keras.layers.Dense(len(class_names), activation='softmax')
])


model_CNN.summary()
model_CNN.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss = keras.losses.SparseCategoricalCrossentropy(),
                  metrics = ['accuracy'])


history = model_CNN.fit(train_images, epochs = 5, validation_data = test_images)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()


#%%
y_pred_probas = model_CNN.predict(test_images)
y_pred = y_pred_probas.argmax(axis = 1)
cm = tf.math.confusion_matrix(tf.concat(list(map(lambda ds : ds[1],test_images)), axis = 0),y_pred)

#%%
def show_confusion_matrix(matrix, labels):
    N = len(labels)
    fig, ax = plt.subplots(figsize=(N,N))
    im = ax.imshow(matrix)
    # We want to show all ticks...
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    plt.show()

show_confusion_matrix(cm, class_names)
