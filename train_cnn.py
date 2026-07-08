import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

IMG_SIZE = (224,224)

BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(

    "dataset",

    validation_split=0.2,

    subset="training",

    seed=42,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)

val_ds = tf.keras.utils.image_dataset_from_directory(

    "dataset",

    validation_split=0.2,

    subset="validation",

    seed=42,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)

# Normalize images

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x,y: (normalization_layer(x),y)
)

val_ds = val_ds.map(
    lambda x,y: (normalization_layer(x),y)
)

# CNN Model

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(224,224,3)
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dense(
        5,
        activation='softmax'
    )

])

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

model.summary()

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=10

)

model.save("cnn_model.keras")

print("Model saved successfully!")

import matplotlib.pyplot as plt

## Accuracy Plot
plt.figure(figsize=(6,4))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()

# Accuracy plot
plt.figure(figsize=(6,4))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()


# Loss plot
plt.figure(figsize=(6,4))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.show()