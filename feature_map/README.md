# DeepHyperion: Feature map

## General Information
This repository contains code to generate rescaled feature maps proposed in the paper "DeepHyperion: Exploring the feature space of Deep Learning systems using Illumination search"

## Docker installation
See the file [docker-install.md](docker-install.md).

## MacOS installation
See the file [macos-install.md](macos-install.md)

## Usage
### Input

* A trained model in h5 format. The default one is in the folder `models`;
* `config.py` containing the configuration of the tool selected by the user.

### Run the Tool


To run the tool use the following command:

```
python feature_map.py
```

### Output

When the run is finished, the tool produces the following outputs in the `logs` folder:

* feature maps representing inputs distribution (in pdf);


## Troubleshooting

* if pip cannot install the correct version of `opencv-python` check whether you upgraded pip correctly after you activate the virtual environment `.venv`

* If tensorflow cannot be installed successfully, try to upgrade the pip version. Tensorflow cannot be installed by old versions of pip. We recommend the pip version 20.1.1.

* If the import of cairo, potrace or other modules fails, check that the correct version is installed. The correct version is reported in the file requirements.txt. The version of a module can be checked with the following command:

```
pip3 show modulename | grep Version
```
    
To fix the problem and install a specific version, use the following command:
    
```
pip3 install 'modulename==moduleversion' --force-reinstall
```


