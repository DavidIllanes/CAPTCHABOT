import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import *
import matplotlib.pyplot as plt


# Figure out class imabalance
# New colors to balance classes
# Try no shuffle image loading
# Figure out val > train accuracy
# Push training to limit
# Optimize params/top network


SEED = 42
IMG_SIZE = (120, 120)
batch_size = 32
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

inception_v3 = inception_v3()

last_layer = inception_v3.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

x = dense_top(last_output, 32, 8, 0.2, 'softmax')

model = Model(inception_v3.input, x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(
      train_generator,
      steps_per_epoch=int(train_generator.n / batch_size),  # images / batch_size if we want to run all images
      epochs=10,
      validation_data=val_generator,
      validation_steps=int(val_generator.n / batch_size))


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
