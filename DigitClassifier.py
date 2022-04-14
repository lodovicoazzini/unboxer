import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class DigitClassifier(Sequential):
    """
    Sequential model to classify digits from the MNIST dataset
    """

    @staticmethod
    def _to_grayscale(images):
        if len(images.shape) > 3 and images.shape[-1] > 1:
            return tf.image.rgb_to_grayscale(images)
        else:
            return images

    def __init__(self):
        super(DigitClassifier, self).__init__([
            layers.Lambda(DigitClassifier._to_grayscale),
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
        self.compile(
            optimizer=Adam(0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()]
        )

    def predict_proba(self, x, batch_size=32, verbose=0):
        """
        Wrapper for predict(). Needed for LIME
        """
        return self.predict(x)
