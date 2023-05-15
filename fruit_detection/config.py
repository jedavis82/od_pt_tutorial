import torch
import os

cache_dir = '../data/models/'
num_classes = 4
model_type = 'fasterrcnn'
model_training = False  # Set to true to train, false to run the test set

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 2
size = (480, 480)  # Resize parameters for the images
target_transform = None  # dataset parameters

# Optimizer params
lr = 0.005
momentum = 0.9
weight_decay = 0.0005
step_size = 3
gamma = 0.1

num_epochs = 10

data_dir = '../data/fruits/'
train_dir = f'{data_dir}train/'
val_dir = f'{data_dir}val/'
test_dir = f'{data_dir}test/'
train_annotations_file = f'{train_dir}annotations.csv'
val_annotations_file = f'{val_dir}annotations.csv'
test_annotations_file = f'{test_dir}annotations.csv'

# Model output params
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_dir = f'{output_dir}models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
metrics_dir = f'{output_dir}metrics/'
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

model_output_file = f'{model_dir}{model_type}_fruit_detector.pt'
train_stats_output_file = f'{metrics_dir}{model_type}_training_stats.csv'
val_stats_output_file = f'{metrics_dir}{model_type}_validation_stats.csv'
test_stats_output_file = f'{metrics_dir}{model_type}_test_stats.csv'
