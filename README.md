# Object Detection Fine Tuning in PyTorch Tutorial
This code base contains a start to finish implementation of fine tuning and object detection model 
using the PyTorch framework. Additionally, after the PyTorch implementation has been created and tested, the 
code base will detail an implementation of setting up the same model using the PyTorch Lightning framework. 

All code was developed on a Windows 11 machine using the PyCharm IDE. 
Unless specified otherwise, installations should be the same across operating systems.
The only exception is Google Colab was used for model training after initial development and verification. 
There are detailed instructions on setting up this environment in the corresponding notebook.

## Installation of Required Tools and Packages 
### Miniconda
All code was developed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Download the appropriate version for your OS and install the tool. 
It is not necessary to use Miniconda, any virtual environment tool will work fine, but the instructions in this 
tutorial are specifically targeted at a Miniconda environment. 
#### Creating your Miniconda Environment 
Open a terminal and execute the following commands: 
```
conda create -n od_pt_tutorial python=3.8
conda activate od_pt_tutorial
```
This will create a Miniconda environment named "od_pt_tutorial" with Python version 3.8.x. 

### Installing Pytorch (GPU version) 
The documentation for installing PyTorch can be found [here](https://pytorch.org/get-started/locally/). 
We will be using Pip installs. To install the GPU version of PyTorch, an extra wheel command must be 
supplied to the pip install command as shown below: 
``` 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
The above example installs PyTorch with Cuda support for Windows. Check the 
[documentation](https://pytorch.org/get-started/locally/) for OS specific install commands. 

### Additional Required Libraries 
The remainder of the required libraries can be installed normally using pip. 
Run the following command from the `<root>/` folder of the repo to activate your conda environment and 
install the required packages: 
``` 
conda activate od_pt_tutorial
pip install -r requirements.txt
```

## Project Structure
Subject to change 
``` 
--od_pt_tutorial
    --data
        --fruits 
            --train
            --test
            --valid
    --training
        --blah 
    --evaluation
        --blah 
    requirements.txt
    .gitignore
    create_dataset.ipynb
```

## PyCharm
The free [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) IDE was used for project development. 
All code should work in VSCode as well. The only caveat are where the scripts execute from 
within the IDEs. VSCode executes from the root of the project directory, whereas PyCharm 
executes from the directory the script is contained in. If any of the code does not work, it is 
worthwhile to verify that there isn't a relative file path issue causing errors. 

## Dataset 
We will be working with a toy [Fruits Dataset](https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/input) from Kaggle. 
The fruits data should be stored in the `od_pt_tutorial/data/fruits/` directory. 
Because of space requirements, the data is not contained in this repo. 

The dataset is not in the proper format for training using PyTorch. 
The `create_dataset.ipynb` notebook can be used for inspecting the data set and formatting it correctly. 
This notebook should be run first to ensure the data is in the proper format, and so that you gain an understanding of 
what PyTorch requires in terms of annotated data. 


