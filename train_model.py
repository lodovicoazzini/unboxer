import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import CLASSIFIER_PATH
from utils.dataset import get_train_test_data

if __name__ == '__main__':
    classifier = Sequential(layers=[
        layers.Lambda(
            lambda images: tf.image.rgb_to_grayscale(images)
            if len(images.shape) > 3 and images.shape[-1] > 1
            else images
        ),
        layers.Conv2D(24, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_1'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(36, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_128'),
        layers.Dense(10, activation='linear', name='visualized')
    ])
    classifier.compile(
        optimizer=Adam(0.001),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()]
    )

    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True, verbose=True)

    # Create an instance of the classifier and train it
    classifier.fit(train_data, train_labels, epochs=1, verbose=True)
    loss, acc = classifier.evaluate(test_data, test_labels)
    print(f'Accuracy on test: {acc}')

    classifier.save(CLASSIFIER_PATH)
