# XAI Techniques Implementations

This document contains report useful information on the techniques used in the project.

| Technique | Resources |
| --- | --- |
| LRP | 
| LIME | [LIME paper resources](https://github.com/marcotcr/lime/tree/master/doc/notebooks)
| SHAP | |
| GRAD-CAM| [XPLIQUE](https://deel-ai.github.io/xplique/) |

## The Model

The model used in this work is a Sequential model from Tensorflow. The model is subclassed so that it can be adapted to the LIME explainer by implementing the required method `predict_proba`.

```python
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


class DigitClassifier(Sequential):
    def __init__(self):
        super(DigitClassifier, self).__init__([
            layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)),
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
        return self.predict(x)

```

## The Techniques

The following is a list of the techniques with some information about how to use them for the particular case of MNIST digits classification.

### LRP

### LIME

LIME requires the images to be in rgb format. You can convert images from grayscale to rgb by using Tensorflow.
```python
import tensorflow as tf

# Turn the images into rgb, expected from LIME
train_data_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(train_data, -1))
test_data_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(test_data, -1))
```

Once the data is in the correct format we can train a classifier.

```python
from DigitClassifier import DigitClassifier

# Create an instance of the classifier and train it
classifier = DigitClassifier()
classifier.fit(
    train_data_rgb,
    train_labels,
    epochs=1
)
```

The explanations for an image can be computed using the `lime` library itself.

```python
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

# Setup the LIME explainer
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=5, ratio=0.2)

# Generate the explanation for an image
explanation = explainer.explain_instance(
    test_data[img_num],
    classifier_fn = classifier.predict_proba,
    top_labels=10, hide_color=0, num_samples=1000, segmentation_fn=segmenter
)
```

### SHAP

### XPLIQUE

