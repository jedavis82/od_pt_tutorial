from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import generalized_box_iou_loss
import torch
import sys
import math
from tqdm import tqdm


def create_model(num_classes):
    # Get the most up-to-date weights for our FasterRCNN pretrained model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # Load our pretrained model using torchvision
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    # Grab the input features from the classification score layer
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    # Create a new FastRCNNPredictor box predictor to work with our classes
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, batch_size, accum_iteration):
    loss_value = 0.0
    model.train()
    lr_scheduler = None
    if epoch == 0:
        # On the first epoch create a linear lr scheduler that will jumpstart the convergence hopefully
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, total=len(data_loader) / batch_size)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Normalize loss to account for batch accumulation
        losses = losses / accum_iteration
        loss_value += losses.item()

        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            print(loss_dict)
            sys.exit(1)

        losses.backward()

        # This is our gradient accumulation logic
        if ((batch_idx + 1) % accum_iteration == 0) or (batch_idx + 1 == len(data_loader)):
            # Push our accumulated gradients through the network and update weights
            optimizer.step()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()
    return loss_value


@torch.inference_mode()
def evaluate(model, data_loader, device, batch_size):
    # TODO: What I'll need to do
    # I'm going to need the loss between the ground truth and predicted boxes which i can get by:
    # generalized_box_iou_loss(outputs[0]['boxes'], targets[0]['boxes'])
    # However the above fails when you have more than one detection per image. For some fucking reason.
    # If I get that working I'll still have to compute the loss for the predicted and output labels.
    # It may just be easier to clone that godforsaken repo

    # The other option is to use this SO answer: https://stackoverflow.com/a/71315672
    # That pretty much reimplements the forward loop of the FRCNN model and accounts for losses
    model.eval()
    cpu_device = torch.device('cpu')
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, total=len(data_loader) / batch_size)):
        images = list(image.to(device) for image in images)
        # targets = list({k: v.to(device) for k, v in t.items()} for t in targets)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # box_loss = generalized_box_iou_loss(outputs[0]['boxes'], targets[0]['boxes'])
        # res = {target['img_id'].item(): output for target, output in zip(targets, outputs)}
        print()