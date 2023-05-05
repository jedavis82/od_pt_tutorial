## Fruit Detection
This package contains the code to format the fruits dataset, create a PyTorch dataset and train an object detection model. 

### Creating Train/Val/Test Splits
The [Kaggle Fruits Dataset](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection) ships with some 
images that raise "known incorrect sRGB warning" messages. To my knowledge, these warnings can be safely ignored. 

Once you have downloaded the fruits dataset from the above link, run the `create_dataset_splits.py` script. 
This script will create a train/val/test split of the data. The original data only comes with a train and test split. 

