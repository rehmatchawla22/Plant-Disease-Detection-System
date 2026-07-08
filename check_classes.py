import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

print(train_ds.class_names)

print("\nNumber of Classes:")

print(len(train_ds.class_names))