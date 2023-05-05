import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import Resize
import pandas as pd
import numpy as np
from typing import Tuple


def fruits_collate_fn(batch):
    """
    A simple collate function for batching our dataset in a dataloader
    :param batch:
    :return:
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
    # return tuple(zip(*batch))


class FruitsDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, size: Tuple[int], transform=None, target_transform=None):
        self.data_df = pd.read_csv(annotations_file, encoding='utf-8', engine='python')
        self.img_dir = img_dir
        self.img_labels = self.data_df['label']
        self.encoding_dict = {'apple': 1, 'banana': 2, 'orange': 3}
        self.img_ids = self.data_df['filename'].unique()  # Use the image file name as the ID for each image
        self.transform = transform
        self.target_transform = target_transform
        target_size = torch.Size(size)
        self.target_x = target_size[1]
        self.target_y = target_size[2]

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]

        # The full file path to the image
        img_path = self.img_dir + img_id

        # Select all rows in our data frame that contain entries for the image at location data_df[index]
        img_annotations = self.data_df.loc[self.data_df['filename'] == img_id]

        # Read the image using torchvision so we can return it
        # FasterRCNN model requires the image to be a floating point type.
        # We use convert_image_dtype to convert from [0-255] to [0.0-1.0]
        image = read_image(img_path, ImageReadMode.RGB)
        image = convert_image_dtype(image, torch.float)

        # Get the original image size. We need to rescale it to work with the required model size
        original_image_size = image.shape
        # Resize the image to match what the model expects
        image = Resize([self.target_x, self.target_y])(image)
        # Get the scale factor for the bounding boxes
        x_scale = self.target_x / original_image_size[1]
        y_scale = self.target_y / original_image_size[2]

        # Get the boxes for our image
        boxes = img_annotations[['x1', 'y1', 'x2', 'y2']].values
        # Resize logic taken from here: https://stackoverflow.com/a/49468149
        boxes[:, 0] = np.rint(boxes[:, 0] * x_scale).astype(int)
        boxes[:, 1] = np.rint(boxes[:, 1] * y_scale).astype(int)
        boxes[:, 2] = np.rint(boxes[:, 2] * x_scale).astype(int)
        boxes[:, 3] = np.rint(boxes[:, 3] * y_scale).astype(int)

        # Get the area of all the boxes. The [:, 3] notation says to give me the entire column at column index 3
        # and so on. This is numpy shorthand for subtracting and multiplying entire columns of arrays
        # The area equation is w*h, and we will have an n-element matrix where each entry is the area of
        # the bounding box for the nth object
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Convert to torch tensors so we can pass to the model easily
        boxes = torch.as_tensor(boxes, dtype=torch.int)
        area = torch.as_tensor(area, dtype=torch.float32)
        # get the labels for our image
        labels = img_annotations['label'].values
        labels = [self.__convert_labels__(x) for x in labels]
        labels = torch.as_tensor(labels)

        # Apply any transforms if they were supplied
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        # Add the results for our image to a dictionary. This dictionary will hold the
        # box, labels, and area of our box.
        # We cannot use the string path for 'img_id' in our targets. Instead we'll use the index passed in
        # to the dataloader
        target = {'boxes': boxes,
                  'area': area,
                  'labels': labels,
                  'img_id': torch.tensor([index])}

        return image, target

    def __convert_labels__(self, x):
        # This function will convert our string labels into integer values
        # Torch will now allow creating a tensor using strings so our workaround will be
        # to use this encoding.
        # Remember, torch reserves 0 for the "background class" so we start at 1
        # TODO: This may not be true for Faster-RCNN, just for Mask-RCNN
        converted_label = self.encoding_dict[x]
        return converted_label

    def __get_encoding_dict__(self):
        return self.encoding_dict
