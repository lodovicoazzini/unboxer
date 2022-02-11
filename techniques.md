# XAI Techniques Implementations

This document contains report useful information on the techniques used in the project.

| Technique | Resources |
| --- | --- |
| LRP | 
| LIME | [LIME paper resources](https://github.com/marcotcr/lime/tree/master/doc/notebooks)
| SHAP | 

### LRP

### LIME

LIME requires the images to be in rgb format. You can convert images from grayscale to rgb by using `numpy` and `scikit-image`.
```python
import numpy as np
from skimage.color import gray2rgb

# Turn the images into rgb, expected from LIME
train_data_rgb = np.array(list(map(gray2rgb, train_data))).astype(np.uint8)
test_data_rgb = np.array(list(map(gray2rgb, test_data))).astype(np.uint8)
```

Once the data is in the right format, you can build a pipeline for the classifier and fit it to the data. The example below is using a `RandomForestClassifier`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier

# Create and fit the pipeline for the random forest classifier
classifier_pipeline = Pipeline([
    ('grayscale', FunctionTransformer(lambda imgs: [rgb2gray(img) for img in imgs])),
    ('flatten', FunctionTransformer(lambda imgs: [img.flatten() for img in imgs])),
    ('RF', RandomForestClassifier())
])

classifier_pipeline.fit(train_data_rgb, train_labels)
```

The explanations for an image can be computed using the `lime` library itself.

```python
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

# Setup the LIME explainer
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

# Generate the explanation for an image
explanation = explainer.explain_instance(
    test_data_rgb[img_num],
    classifier_fn = classifier_pipeline.predict_proba,
    top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter
)
```

### SHAP