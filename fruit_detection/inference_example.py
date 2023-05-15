import torch
import config
from model_utils import get_od_model, apply_nms, torch_to_pil, get_transforms
from fruit_dataset import FruitDataset, fruits_collate_fn
from torch.utils.data import DataLoader
import cv2
import numpy as np


def display_results(img, boxes, labels):
    """
    Accept the tensor results, and convert the tensors appropriately for displaying detection output
    :param img: Tensor representation of an image
    :param boxes: Tensor of bounding boxes for an image
    :param labels: Tensor of labels for an image
    :return: None
    """
    encoding_dict = {1: 'apple', 2: 'banana', 3: 'orange'}
    image = img.detach().numpy()  # Detach the image from the GPU and convert it to an NDArray
    # OpenCV expects images in (Channels, Width, Height) format. Torch output is (Width, Height, Channels)
    image = image.transpose(1, 2, 0)
    # Convert color ranges from [0-1] to [0-255]
    image = np.multiply(image, 255).astype(np.uint8)
    # Convert the color from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the boxes and labels on the image
    for b, l in zip(boxes, labels):
        box = b.cpu().numpy().astype(np.int32)
        label = encoding_dict[l.item()]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        cv2.putText(image, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
        print()
    # Display the image
    cv2.imshow('Output', image)
    cv2.waitKey(0)


def main():
    model = get_od_model(4)
    model.load_state_dict(torch.load(config.model_output_file))
    model.eval()
    model.to(config.device)
    test_dataset = FruitDataset(
        annotations_file=config.test_annotations_file, img_dir=config.test_dir,
        size=config.size,
        transform=get_transforms(train=False),
        target_transform=config.target_transform
    )
    test_loader = DataLoader(test_dataset, 1, shuffle=True, collate_fn=fruits_collate_fn)

    # Pull one sample from the test dataset
    # img, target = test_dataset[5]
    # Loop through the test dataset to verify it's working
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img, target = test_dataset[idx]
            prediction = model([img.to(config.device)])[0]
            nms_prediction = apply_nms(prediction)
            boxes = nms_prediction['boxes']
            labels = nms_prediction['labels']
            display_results(img, boxes, labels)


if __name__ == '__main__':
    main()
