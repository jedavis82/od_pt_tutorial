import os
import torch
from fruits_dataset import FruitsDataset, fruits_collate_fn
from torch.utils.data import DataLoader
from utils import create_model, train_one_epoch, evaluate


# Let's create a dataset and dataloader from our training directory images/annotations
# TODO: Make these args you can pass in
training_img_dir = './data/fruits/train/'
training_annotations_file = './data/fruits/train/annotations.csv'
val_img_dir = './data/fruits/valid/'
val_annotations_file = './data/fruits/valid/annotations.csv'
cache_dir = './data/models/'
os.environ['TORCH_HOME'] = cache_dir
num_epochs = 10
batch_size = 1


def main():
    # Create the model to use for training. Remember we have to account for the background class
    od_model = create_model(num_classes=4)

    # Create our optimizer and learning rate scheduler
    params = [p for p in od_model.parameters() if p.requires_grad]
    # Define our AdamW optimizer with a higher learning rate to start
    lr = 1e-3
    optimizer = torch.optim.AdamW(params, lr=lr)
    # Define our learning rate scheduler that will decrease the lr as training runs
    step_size = 3  # Number of epochs to run before decaying the learning rate
    gamma = 0.1  # The multiplicative value to reduce the learning rate by. We'll use 10%
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    # Create our datasets and dataloaders
    train_dataset = FruitsDataset(training_annotations_file, training_img_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fruits_collate_fn)
    val_dataset = FruitsDataset(val_annotations_file, val_img_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=fruits_collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # Place our model on the GPU if available
    od_model.to(device)

    # We will use gradient accumulation to simulate a larger batch size
    # Due to size constraints on GPU memory, we currently can only handle a batch size of 1 on
    # my laptop's GPU. Using gradient accumulation and simulating a larger batch size, we can
    # prevent our weight updates from occurring too frequently.
    accum_iteration = 32

    for epoch in range(num_epochs):
        # train_loss = train_one_epoch(od_model, optimizer, train_dataloader, device, epoch, batch_size, accum_iteration)
        # print(f'Epoch {epoch}: Loss: {train_loss}')

        # Update the learning rate
        # lr_scheduler.step()
        # Evaluate on the val dataset
        evaluate(od_model, val_dataloader, device, batch_size)


if __name__ == '__main__':
    main()
