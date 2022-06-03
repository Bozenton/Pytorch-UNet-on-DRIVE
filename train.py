import logging
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.DriveDataset import DriveDataset
from unet import UNet
from utils.dice_score import dice_loss
from evaluate import evaluate

dir_img = Path('./datasets/training/images')
# train_label_dir = Path('./datasets/training/1st_manual')
dir_mask = Path('./datasets/training/1st_manual')
dir_checkpoint = Path('./checkpoints/')

writer = SummaryWriter()

def train(net: UNet,
          device,
          epochs: int = 5,
          batch_size: int = 1,
          learning_rate: float = 1e-5,
          val_percent: float = 0.1,
          save_checkpoint: bool = True,
          img_scale: float = 1,
          amp: bool = False):
    dataset = DriveDataset(dir_img, dir_mask, img_scale,
                           mask_suffix='_manual1')

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=True, **loader_args)

    # experiment = wandb.init(project='UNetDRIVE',
    #                         resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))
    logging.info(f'''Starting training:
      Epochs:          {epochs}
      Batch size:      {batch_size}
      Learning rate:   {learning_rate}
      Training size:   {n_train}
      Validation size: {n_val}
      Checkpoints:     {save_checkpoint}
      Device:          {device.type}
      Images scaling:  {img_scale}
      Mixed Precision: {amp}
  ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    train_steps = len(train_loader)

    # Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'] # b, 3, 584, 565
                masks = batch['mask']   # b, 584, 565
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images) # b, 1, 584, 565
                    loss = criterion(masks_pred, masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(masks, net.n_classes).permute(
                                        0, 3, 1, 2).float(),
                                    multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' +
                        #                tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' +
                        #                tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info(
                            'Validation Dice score: {}'.format(val_score))
                        
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     'images': wandb.Image(images[0].cpu()),
                        #     'masks': {
                        #         'true': wandb.Image(masks[0].float().cpu()),
                        #         'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        #     },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })
                        writer.add_scalar(tag='learning_rate', scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)
                        writer.add_scalar(tag='val_score', scalar_value=val_score, global_step=global_step)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint /
                       'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.to(device=device)

    weight_file = './checkpoints/checkpoint_epoch50.pth'
    if os.path.exists(weight_file):
        net.load_state_dict(torch.load(weight_file, map_location=device))
        logging.info(f'load pre-trained weights')

    logging.info(f'Network:\n'
                f'\t{net.n_channels} input channels\n'
                f'\t{net.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    try:
        train(net=net,
            epochs=100,
            batch_size=4,
            learning_rate=1e-6,
            device=device,
            img_scale=1,
            val_percent=0.2,
            amp=False)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
