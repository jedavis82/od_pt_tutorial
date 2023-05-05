## PyTorch and PyTorch Lightning Object Detection 
This repo contains several scripts for training object detection models on various datasets. 
Additionally, this repo contains scripts to leverage PyTorch lightning for model training.

### Installing Requirements
This project was developed using Python 3.9.x in an Anaconda environment. To install the required packages run the
following command from the terminal:

`pip install -r requirements.txt`

This will install PyTorch 2.0 with CUDA support from the `torch_requirements.txt` file as well. To verify the 
installation was successful, launch python from the terminal and run the following commands: 

```
import torch
torch.__version__
torch.cuda.is_available()
```

### Torchvision Leveraged Scripts
For training, some reference scripts from the [Official Torchvision Repo](https://github.com/pytorch/vision) were 
used. These scripts are included in this repo in the `tvision_utils` package. The original location of these files 
from the Torchvision repo are in the `./vision/references/detection/` folder. 

### Datasets
This repo contains training and inference code that works on several data sets.
1. [Kaggle Fruits Dataset](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection)
2. [Roboflow Dollar Bill Detection](https://universe.roboflow.com/alex-hyams-cosqx/dollar-bill-detection)
3. [Roboflow Pistols Detection](https://public.roboflow.com/object-detection/pistols)
4. [Roboflow License Plates](https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn)
5. [Roboflow Vehicles-OpenImages](https://public.roboflow.com/object-detection/vehicles-openimages)

Each of these datasets should be placed in their own folder under the `./data/` directory in this repo. If any 
preprocessing scripts should be run on the datasets, instructions will be given in the accompanying training 
README under each model package. 

### Models 
A PyTorch and PyTorch lightning model will be trained on each of the datasets listed above. Torch and Torch lightning 
will generate a model in the same format, but using Torch Lightning makes constructing the training scripts much 
easier. The end result for this repo is to construct a notebook that guides the user through the process of 
performing transfer learning on a pretrained object detection model for custom classes. 