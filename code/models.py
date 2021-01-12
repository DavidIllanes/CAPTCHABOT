import tensorflow as tf
from tensorflow.keras import layers


def inception_v3(input_shape=(120, 120, 3), include_top=False):
    pre_trained_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=include_top,
                                                          weights='imagenet')
    for layer in pre_trained_model.layers:
        layer.trainable = False
    return pre_trained_model


def dense_top(input_layer, num_neurons=32, num_outputs=8, dropout=0.2, activation_output='softmax'):
    x = layers.Flatten()(input_layer)
    x = layers.Dense(num_neurons, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_outputs, activation=activation_output)(x)
    return x
