import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from config.config_dirs import CLASSIFIER_PATH, PREDICTIONS_PATH
from utils.dataset import get_train_test_data


def create_model():
    print('Creating the classifier ...')

    # Set up the sequential classifier
    classifier = Sequential(layers=[
        layers.Lambda(
            lambda images: tf.image.rgb_to_grayscale(images)
            if len(images.shape) > 3 and images.shape[-1] > 1
            else images
        ),
        layers.Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(.5),
        layers.Flatten(),
        layers.Dense(128),
        layers.Activation('relu'),
        layers.Dropout(.5),
        layers.Dense(10),
        layers.Activation('softmax')
    ])
    classifier.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

    # Get the train test data and convert the labels to categorical
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True, verbose=True)
    train_labels_cat, test_labels_cat = to_categorical(train_labels, 10), to_categorical(test_labels, 10)

    # Train the classifier
    classifier.fit(
        train_data,
        train_labels_cat,
        epochs=50,
        batch_size=218,
        shuffle=True,
        verbose=True,
        validation_data=(test_data, test_labels_cat)
    )
    loss, acc = classifier.evaluate(test_data, test_labels_cat)
    print(f'Accuracy on test: {acc}')

    # Save the classifier
    classifier.save(CLASSIFIER_PATH)

    return classifier


def generate_predictions(classifier, test_data):
    print('Generating the predictions ...')
    # Compute the predictions
    predictions = classifier.predict(test_data).argmax(axis=-1)
    # Save the predictions
    np.savetxt(PREDICTIONS_PATH, predictions, delimiter=',')

    return predictions
