import timm
import os
from PIL import Image
import torch
import numpy as np
from od_model import FruitDetector
from fruits_dataset import FruitsDataset, fruits_collate_fn
from torch.utils.data import DataLoader

base_model_name = 'resnet50d'
num_classes = 3
cache_dir = './data/models/'

training_img_dir = './data/fruits/train/'
training_annotations_file = './data/fruits/train/annotations.csv'
sample_file = './data/fruits/train/apple_3.jpg'


def main():
    os.environ['TORCH_HOME'] = cache_dir
    # Create a base model using the timm library.
    base_model = timm.create_model(base_model_name, pretrained=True)

    # Extract the default image size from the model config. This will be used to resize our images.
    input_size = base_model.default_cfg['input_size']
    model = FruitDetector(base_model, num_classes)
    # Check the params here
    params = [p for p in model.parameters() if p.requires_grad]

    train_dataset = FruitsDataset(training_annotations_file, training_img_dir, input_size)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=fruits_collate_fn)

    # # base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)
    # base_model = timm.create_model(base_model_name, pretrained=True)
    sample_image = Image.open(sample_file)
    sample_image = torch.as_tensor(np.array(sample_image, dtype=np.float32))[None]#.transpose(2, 0)[None]
    # # features = base_model.forward_features(sample_image)
    # features = base_model(sample_image)
    # print()

    for batch_idx, (images, targets) in enumerate(train_dataloader):
        # images = list(images)
        features = model(images)
        # TODO: This is working for using the model as a feature extractor. I need to flesh out the rest of the
        # the model code (regress and class head) and run data through there to see if it's working properly

if __name__ == '__main__':
    main()
