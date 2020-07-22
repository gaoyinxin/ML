import sys

import numpy as np
import tensorflow as tf
from absl import logging
from tensorflow.keras import datasets, layers, utils


def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    training_size = len(train_images)
    test_size = len(test_images)
    train_images = np.asarray(train_images, dtype=np.float32) / 255
    train_images = train_images.reshape((training_size, 28, 28, 1))
    test_images = np.asarray(test_images, dtype=np.float32) / 255
    test_images = test_images.reshape((test_size, 28, 28, 1))
    train_labels = utils.to_categorical(train_labels, 10)
    test_labels = utils.to_categorical(test_labels, 10)
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    return (train_images, train_labels), (test_images, test_labels)


def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.summary()
    return model


def convert_model_2_estimator(model):
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])
    strategy = None
    # https://zhuanlan.zhihu.com/p/73580663
    # strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy, keep_checkpoint_max=5,
                                    log_step_count_steps=20, save_checkpoints_steps=200)
    return tf.keras.estimator.model_to_estimator(model, config=config)


def input_fn(images, labels, epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    shuffle_size = 5000
    dataset = dataset.shuffle(shuffle_size).repeat(epochs).batch(batch_size)
    dataset = dataset.prefetch(None)
    return dataset


def func():
    logging.set_verbosity(logging.INFO)

    (train_images, train_labels), (test_images, test_labels) = load_data()

    model = build_model()

    estimator = convert_model_2_estimator(model)

    batch_size = 512
    estimator_train_result = estimator.train(input_fn=lambda: input_fn(train_images, train_labels, epochs=10, batch_size=batch_size))
    print(estimator_train_result)
    score = estimator.evaluate(lambda: input_fn(test_images, test_labels, epochs=1, batch_size=batch_size))
    print(score)


if __name__ == "__main__":
    func()
    sys.exit()
