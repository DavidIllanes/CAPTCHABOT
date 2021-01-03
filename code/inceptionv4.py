import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TODO figure out class imbalance
# TODO check sigmoid to see uncertainty at output

SEED = 42
IMG_SIZE = (120, 120)
train_dir = '../data/train'
val_dir = '../data/val'
test_dir = '../data/test'

# Generator that adds noise to training images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Flows to read data in batches
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(120, 120), batch_size=32,
                                                    class_mode='categorical', shuffle=True, seed=SEED)

val_generator = val_datagen.flow_from_directory(val_dir, target_size=(120, 120), batch_size=32,
                                                class_mode='categorical', shuffle=True, seed=SEED)

IMG_SHAPE = IMG_SIZE + (3,)

pre_trained_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(8, activation='softmax')(x)

model = Model(pre_trained_model.input, x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=2,
      validation_data=val_generator,
      validation_steps=50,
      verbose=2)
