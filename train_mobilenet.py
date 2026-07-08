import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import json

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# Load Dataset
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

# Save class names
class_names = train_ds.class_names

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("\nNumber of Classes:", len(class_names))
print("\nClass Names:\n")

for c in class_names:
    print(c)

# Normalize
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# Load MobileNetV2
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze pretrained layers
base_model.trainable = False

# Number of classes
num_classes = len(class_names)

# Build model
model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(0.3),

    layers.Dense(
        num_classes,
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

model.save("models/multicrop_mobilenetv2.keras")

print("\nModel saved successfully!")