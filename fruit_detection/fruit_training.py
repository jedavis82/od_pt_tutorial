import torch
from torch.utils.data import DataLoader
import pandas as pd
import config
from fruit_dataset import FruitDataset, fruits_collate_fn
from tvision_utils.engine import train_one_epoch, evaluate
from model_utils import get_od_model, get_transforms


def main():
    if config.model_training:
        model = get_od_model(config.num_classes, config.model_type)
        model.to(config.device)

        # Set up optimizer and learning rate scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        # LR scheduler that decreases the learning rate by 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )

        train_dataset = FruitDataset(
            annotations_file=config.train_annotations_file,
            img_dir=config.train_dir,
            size=config.size,
            transform=get_transforms(train=True),
            target_transform=config.target_transform
        )
        train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, collate_fn=fruits_collate_fn)

        val_dataset = FruitDataset(
            annotations_file=config.val_annotations_file,
            img_dir=config.val_dir,
            size=config.size,
            transform=get_transforms(train=False),
            target_transform=config.target_transform
        )
        val_loader = DataLoader(val_dataset, config.batch_size, shuffle=True, collate_fn=fruits_collate_fn)

        # We'll store the averaged loss over the box regression and label classification for each epoch
        train_loss = []
        # We'll store the mAP values for varying IOU thresholds each epoch
        map_metrics = []
        # We'll use the mAP @ IOU = 0.75 to determine when we need to save our model
        best_map_val = 0.0
        for epoch in range(config.num_epochs):
            train_metrics = train_one_epoch(model, optimizer, train_loader, config.device, epoch, print_freq=10)
            lr_scheduler.step()
            eval_metrics = evaluate(model, val_loader, device=config.device)

            # The combined loss between box regression and classification (SmoothedValue object)
            train_loss_value = train_metrics.meters['loss'].value
            train_loss.append({'epoch': epoch, 'loss': train_loss_value})

            # Extract the mAP @ IOU=0.5 value from evaluation
            map_iou_05 = eval_metrics.coco_eval['bbox'].stats[1]
            # Extract the mAP @ IOU=0.75 value from evaluation
            map_iou_075 = eval_metrics.coco_eval['bbox'].stats[2]
            # Extract the mAP @ IOU=0.5:0.95 value from evaluation
            map_iou_05_095 = eval_metrics.coco_eval['bbox'].stats[0]
            map_metrics.append({
                'epoch': epoch,
                'map@iou=0.5': map_iou_05,
                'map@iou=0.75': map_iou_075,
                'map@iou=0.5:0.95': map_iou_05_095
            })

            curr_map = map_iou_075
            if curr_map > best_map_val:
                print(f'Saving new best model. mAP: {curr_map}')
                best_map_val = curr_map
                # Save the model here
                torch.save(model.state_dict(), config.model_output_file)
        train_df = pd.DataFrame.from_dict(train_loss)
        val_df = pd.DataFrame.from_dict(map_metrics)
        train_df.to_csv(config.train_stats_output_file, header=True, index=False, encoding='utf-8')
        val_df.to_csv(config.val_stats_output_file, header=True, index=False, encoding='utf-8')
    else:
        # Test the model
        model = get_od_model(config.num_classes, config.model_type)
        model.load_state_dict(torch.load(config.model_output_file))
        model = model.to(config.device)

        test_dataset = FruitDataset(
            annotations_file=config.test_annotations_file,
            img_dir=config.test_dir,
            size=config.size,
            transform=get_transforms(train=False),
            target_transform=config.target_transform
        )
        test_loader = DataLoader(test_dataset, config.batch_size, shuffle=True, collate_fn=fruits_collate_fn)
        test_metrics = evaluate(model, test_loader, device=config.device)
        map_iou_05 = test_metrics.coco_eval['bbox'].stats[1]
        map_iou_075 = test_metrics.coco_eval['bbox'].stats[2]
        map_iou_05_095 = test_metrics.coco_eval['bbox'].stats[0]

        test_map_metrics = {
            'map@iou=0.5': map_iou_05,
            'map@iou=0.75': map_iou_075,
            'map@iou=0.5:0.95': map_iou_05_095
        }
        test_df = pd.DataFrame.from_dict([test_map_metrics])
        test_df.to_csv(config.test_stats_output_file, header=True, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
