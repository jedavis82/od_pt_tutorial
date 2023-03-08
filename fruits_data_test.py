from fruits_dataset import FruitsDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import cv2

train_img_dir = './data/fruits/train/'
train_annotations_file = './data/fruits/train/annotations.csv'


def collate_fn(batch):
    return tuple(zip(*batch))


def show(imgs):
    # There are corrupted files in the fruits dataset. This should work for any other valid data though.
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_results(img_name, boxes, labels):
    img_path = train_img_dir + img_name
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for box, label in zip(boxes, labels):
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.imshow('Annotations', img)
    cv2.waitKey(0)


# This will test out the fruits dataset and make sure we can load samples etc.
def main():
    # Create an instance of our dataset class and test it with a data loader
    train_dataset = FruitsDataset(train_annotations_file, train_img_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=10,
                                  shuffle=True, collate_fn=collate_fn)

    train_images, train_targets = next(iter(train_dataloader))
    # PyTorch does not allow "writing text" on an image the same way OpenCV does
    # So we'll make a dictionary mapping for our classes to correspond to a certain color
    # Classes are 1: apple, 2: banana, 3: orange, 4: mixed
    colors_dict = {1: 'red', 2: 'yellow', 3: 'orange', 4: 'blue'}
    cv_labels_dict = {1: 'apple', 2: 'banana', 3: 'orange', 4: 'mixed'}
    for img, targ in zip(train_images, train_targets):
        boxes = targ['boxes']
        labels = targ['labels']
        print(targ['img_id'])
        colors = [colors_dict[int(l)] for l in labels]
        cv_labels = [cv_labels_dict[int(l)] for l in labels]
        show_results(targ['img_id'], boxes.tolist(), cv_labels)
        # This would normally work if there weren't corrupted images
        # result = draw_bounding_boxes(img, boxes, colors=colors, width=2)
        # show(result)
    print()


if __name__ == '__main__':
    main()
