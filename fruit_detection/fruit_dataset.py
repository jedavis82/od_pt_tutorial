"""
This script houses the Torch Dataset class for our fruits data.
It will be used with a dataloader in order to perform model training.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple
import cv2


def fruits_collate_fn(batch):
    """
    A simple collate function for batching our dataset in a dataloader
    :param batch:
    :return:
    """
    # images = torch.stack([item[0] for item in batch])
    # targets = [item[1] for item in batch]
    # return images, targets
    return tuple(zip(*batch))


class FruitDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, size: Tuple[int] = None,
                 transform=None, target_transform=None):
        self.data_df = pd.read_csv(annotations_file, encoding='utf-8', engine='python')
        self.img_dir = img_dir
        self.img_labels = self.data_df['label']
        self.encoding_dict = {'apple': 1, 'banana': 2, 'orange': 3}
        self.img_ids = self.data_df['filename'].unique()  # Use the image file name as the ID for each image
        self.transform = transform
        self.target_transform = target_transform
        if size is not None:
            target_size = torch.Size(size)
            self.width = target_size[0]
            self.height = target_size[1]
        else:
            self.width = None
            self.height = None

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]

        # The full file path to the image
        img_path = self.img_dir + img_id

        # Select all rows in our data frame that contain entries for the image at location data_df[index]
        img_annotations = self.data_df.loc[self.data_df['filename'] == img_id]

        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Image shape is height x width in cv2
        original_image_size = img.shape
        orig_width = original_image_size[1]
        orig_height = original_image_size[0]
        if self.width is not None and self.height is not None:
            image = cv2.resize(img_rgb, (self.width, self.height))
            # Divide by 255 to get values in the range 0-1 instead of 0-255
            image /= 255.0
        else:
            # Just do the /255 and do not resize
            img_rgb /= 255.0
            image = img_rgb

        # Get the boxes for our image
        boxes = img_annotations[['x0', 'y0', 'x1', 'y1']].values
        if self.width is not None and self.height is not None:
            boxes[:, 0] = (boxes[:, 0]/orig_width)*self.width
            boxes[:, 1] = (boxes[:, 1]/orig_height)*self.height
            boxes[:, 2] = (boxes[:, 2]/orig_width)*self.width
            boxes[:, 3] = (boxes[:, 3]/orig_height)*self.height

        # Get the area of all the boxes. The [:, 3] notation says to give me the entire column at column index 3
        # and so on. This is numpy shorthand for subtracting and multiplying entire columns of arrays
        # The area equation is w*h, and we will have an n-element matrix where each entry is the area of
        # the bounding box for the nth object
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Convert to torch tensors so we can pass to the model easily
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)

        # get the labels for our image
        labels = img_annotations['label'].values
        labels = [self.__convert_labels__(x) for x in labels]
        labels = torch.as_tensor(labels)

        # Add the results for our image to a dictionary. This dictionary will hold the
        # box, labels, and area of our box.
        # We cannot use the string path for 'image_id' in our targets. Instead we'll use the index passed in
        # to the dataloader
        # We also need a 'iscrowd' key when using the torchvision engine.evaluate()

        target = {'boxes': boxes,
                  'area': area,
                  'labels': labels,
                  'image_id': torch.tensor([index]),
                  'iscrowd': torch.zeros((boxes.shape[0], ), dtype=torch.int64)
                  }

        if self.transform:
            sample = self.transform(image=np.array(image), bboxes=target['boxes'], labels=labels)
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

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

