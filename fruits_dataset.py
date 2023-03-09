import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms.functional import convert_image_dtype
import pandas as pd


def fruits_collate_fn(batch):
    """
    A simple collate function for batching our dataset in a dataloader
    :param batch:
    :return:
    """
    return tuple(zip(*batch))


class FruitsDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_transform=None):
        self.data_df = pd.read_csv(annotations_file, encoding='utf-8', engine='python')
        self.img_dir = img_dir
        self.img_labels = self.data_df['label']
        self.img_ids = self.data_df['filename'].unique()  # Use the image file name as the ID for each image
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]

        # The full file path to the image
        img_path = self.img_dir + img_id

        # Select all rows in our data frame that contain entries for the image at location data_df[index]
        img_annotations = self.data_df.loc[self.data_df['filename'] == img_id]

        # Get the boxes and labels for our image
        # Convert them to torch tensors so we can use them in our dataloader
        boxes = img_annotations[['x1', 'y1', 'x2', 'y2']].values
        # Get the area of all the boxes. The [:, 3] notation says to give me the entire column at column index 3
        # and so on. This is numpy shorthand for subtracting and multiplying entire columns of arrays
        # The area equation is w*h, and we will have an n-element matrix where each entry is the area of
        # the bounding box for the nth object
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        boxes = torch.as_tensor(boxes, dtype=torch.int)
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = img_annotations['label'].values
        labels = [self.__convert_labels__(x) for x in labels]
        labels = torch.as_tensor(labels)

        # Read the image using torchvision so we can return it
        # FasterRCNN model requires the image to be a floating point type.
        # We use convert_image_dtype to convert from [0-255] to [0.0-1.0]
        image = read_image(img_path, ImageReadMode.RGB)
        image = convert_image_dtype(image, torch.float)

        # Apply any transforms if they were supplied
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        # Add the results for our image to a dictionary. This dictionary will hold the
        # box, labels, and area of our box
        target = {'boxes': boxes,
                  'area': area,
                  'labels': labels,
                  'img_id': img_id}

        return image, target

    def __convert_labels__(self, x):
        # This function will convert our string labels into integer values
        # Torch will now allow creating a tensor using strings so our workaround will be
        # to use this encoding.
        # Remember, torch reserves 0 for the "background class" so we start at 1
        # TODO: This may not be true for Faster-RCNN, just for Mask-RCNN
        encoding = {'apple': 1, 'banana': 2, 'orange': 3}
        converted_label = encoding[x]
        return converted_label
