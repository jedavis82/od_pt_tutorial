import math
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights, RetinaNetClassificationHead
from torchvision.models.detection.fcos import FCOSClassificationHead, FCOS_ResNet50_FPN_Weights
from torchvision import transforms as torchtrans
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import config

# Set the torch home directory so that all models will be downloaded there
if not os.path.exists(config.cache_dir):
    os.makedirs(config.cache_dir)
os.environ['TORCH_HOME'] = config.cache_dir


def get_od_model(num_classes, model_type):
    # Model type should be one of: fasterrcnn, retinanet, fcos
    if model_type.lower() == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        # Get the number of input features of the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pretrained head with a new one we will fine tune
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    elif model_type.lower() == 'retinanet':
        # Taken from this SO answer: https://datascience.stackexchange.com/a/96815
        model = torchvision.models.detection.retinanet.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes
        cls_logits = torch.nn.Conv2d(in_features, num_anchors*num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1-0.01)/0.01))
        model.head.classification_head.cls_logits = cls_logits
        return model
    elif model_type.lower() == 'fcos':
        model = torchvision.models.detection.fcos.fcos_resnet50_fpn(FCOS_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            num_classes=num_classes, num_anchors=num_anchors, in_channels=in_features
        )
        return model
    else:
        raise Exception("Invalid model type supplied")


def get_transforms(train=False):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0)  # Converts image to pytorch tensor without /255
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def apply_nms(pred, iou_threshold=0.3):
    keep = torchvision.ops.nms(pred['boxes'], pred['scores'], iou_threshold)
    final_pred = pred
    final_pred['boxes'] = final_pred['boxes'][keep]
    final_pred['scores'] = final_pred['scores'][keep]
    final_pred['labels'] = final_pred['labels'][keep]

    return final_pred


def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')
