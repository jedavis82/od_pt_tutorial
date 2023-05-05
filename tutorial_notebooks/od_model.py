"""
This is our vanilla PyTorch object detection model code.
It will use Resnet50 as a backbone and we'll add in the appropriate layers for label classification and
bounding box regression.
"""

import timm
import torch
from torch import nn
import os


class FruitDetector(nn.Module):
    def __init__(self, base_model=None, num_classes=None):
        super(FruitDetector, self).__init__()
        self.base_model = base_model

        # Assign our number of classes for the model
        self.num_classes = num_classes

        # Now we need to get all but the classification head of the base model
        # We'll be extending it with our own classifier head and box regression head
        # To do so, (experiment) we will get the named classifier head and replace it with the Identity matrix
        # so that we just get the features as output from our base model

        # Get the base classifier name so we can replace it
        base_classifier_name = self.base_model.default_cfg['classifier']
        # Get the number of input features to the base model classifier
        base_model_input_features = self.base_model.get_classifier().in_features
        # Replace the base model classifier with the Identity() matrix
        # self.base_model.base_classifier_name = nn.Identity()

        # # Can get all layers but the classification head this way too
        # base_model_layers = list(self.base_model.children())
        # base_model_num_layers = len(base_model_layers)
        # base_model_layers = base_model_layers[:base_model_num_layers - 2]
        # self.base_model = nn.Sequential(*base_model_layers)

        # Create the updated box regression and classification heads
        # Need to pass in the base model input features to each head and then downsample in layers

    def forward(self, x):
        # Just testing for now. Trying to determine what is output here
        features = self.base_model(x)
        return features



