import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
from torchmetrics import Accuracy, Precision, Recall, Specificity, ROC, Dice

from utils.DriveDataset import DriveDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.metrics import get_accuracy, get_specificity, get_sensitivity, BinaryConfusionMatrix
from unet import UNet

from tqdm import tqdm

img_dir = Path('./datasets/test/images')
gt_dir = Path('./datasets/test/manual')  # ground truth
out_dir = Path('./datasets/test/predict')
assert os.path.exists(img_dir), "{} does not exist".format(img_dir)
assert os.path.exists(gt_dir), "{} does not exist".format(gt_dir)
assert os.path.exists(out_dir), "{} does not exist".format(out_dir)

batch_size = 4

if __name__ == '__main__':
    test_dataset = DriveDataset(img_dir, gt_dir, scale=1, mask_suffix='_manual1')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    weight_file = './checkpoints/checkpoint_epoch100.pth'
    assert os.path.exists(weight_file), "Weight file {} does not exist, " \
                                        "you can download it from ...".format(weight_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.to(device=device)
    net.load_state_dict(torch.load(weight_file, map_location=device))
    net.eval()
    num_test_batches = len(test_loader)

    accuracy = Accuracy(threshold=0.0, num_classes=2, average='macro', mdmc_average='samplewise')
    precision = Precision(threshold=0.0, num_classes=2, average='macro', mdmc_average='samplewise')
    recall = Recall(threshold=0.0, num_classes=2, average='macro', mdmc_average='samplewise')
    specificity = Specificity(threshold=0.0, num_classes=2, average='macro', mdmc_average='samplewise')
    dice = Dice(threshold=0.0, num_classes=2, average='macro', mdmc_average='samplewise')
    roc = ROC(num_classes=2)

    dice_score = 0
    for batch in tqdm(test_loader,
                    total=num_test_batches,
                    desc='Test round', unit='batch', leave=False):
        image_name = batch['name']
        image = batch['image']  # torch.Size([2, 3, 584, 565]) BCHW
        gt = batch['mask']  # torch.Size([2, 584, 565]) BHW
        gt_for_dice_score = F.one_hot(gt, net.n_classes).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            mask_pred = net(image.to(device))  # torch.Size([1, 2, 584, 565]) BCHW
            
            mask_for_dice_score = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(mask_for_dice_score[:, 1:, ...], 
                                    gt_for_dice_score[:, 1:, ...], 
                                    reduce_batch_first=False)

            batch_acc = accuracy(mask_pred.to(device), gt.to(device))
            batch_prc = precision(mask_pred.to(device), gt.to(device))
            batch_rcl = recall(mask_pred.to(device), gt.to(device))
            batch_spc = specificity(mask_pred.to(device), gt.to(device))
            batch_dic = dice(mask_pred.to(device), gt.to(device))
            batch_roc = roc(mask_pred.to(device), gt.to(device))
            

    print(accuracy.compute(), precision.compute(), recall.compute(), specificity.compute(), dice.compute())
    fpr, tpr, thresholds = roc.compute()
    # tensor(0.8557, device='cuda:0') tensor(0.8962, device='cuda:0') tensor(0.8557, device='cuda:0') tensor(0.8557, device='cuda:0') tensor(0.8707, device='cuda:0')   
    # print(dice_score/num_test_batches)

    # TNRfont = {'fontname':'Times New Roman'}
    TNRfont = {'fontname':'DejaVu Sans'}
    fig, ax = plt.subplots(figsize=[5,5], dpi=300, layout='constrained')  # a figure with a single Axes
    ax.plot(fpr[1].cpu().numpy(), tpr[1].cpu().numpy())
    ax.set_xlabel('False Positive Rate', **TNRfont)
    ax.set_ylabel('True Positive Rate', **TNRfont)
    ax.set_title('ROC Curve', **TNRfont)
    plt.savefig('roc_curve.png', format='png', bbox_inches='tight', transparent=True, dpi=300)

    prob = mask_pred[:, 1, ...].cpu()
    assert prob.shape == gt.shape
    # draw some example
    prob_sigmoid = torch.sigmoid(prob) > 0.4
    prob_sigmoid = prob_sigmoid.float()

    fig = plt.figure(figsize=[10,10], dpi=300)

    ax11 = fig.add_subplot(3,3,1)
    ax11.imshow(image[0, ...].permute(1,2,0).numpy())
    ax11.set_axis_off()
    ax11.set_title('Original Image', **TNRfont)

    ax12 = fig.add_subplot(3,3,2)
    ax12.imshow(gt[0, ...].numpy(), cmap='gray')
    ax12.set_axis_off()
    ax12.set_title('Ground Truth', **TNRfont)

    ax13 = fig.add_subplot(3,3,3)
    ax13.imshow(prob_sigmoid[0, ...].numpy(), cmap='gray')
    ax13.set_axis_off()
    ax13.set_title('UNet Output', **TNRfont)

    ax21 = fig.add_subplot(3,3,4)
    ax21.imshow(image[1, ...].permute(1,2,0).numpy())
    ax21.set_axis_off()

    ax22 = fig.add_subplot(3,3,5)
    ax22.imshow(gt[1, ...].numpy(), cmap='gray')
    ax22.set_axis_off()

    ax23 = fig.add_subplot(3,3,6)
    ax23.imshow(prob_sigmoid[1, ...].numpy(), cmap='gray')
    ax23.set_axis_off()

    ax31 = fig.add_subplot(3,3,7)
    ax31.imshow(image[3, ...].permute(1,2,0).numpy())
    ax31.set_axis_off()

    ax32 = fig.add_subplot(3,3,8)
    ax32.imshow(gt[3, ...].numpy(), cmap='gray')
    ax32.set_axis_off()

    ax33 = fig.add_subplot(3,3,9)
    ax33.imshow(prob_sigmoid[3, ...].numpy(), cmap='gray')
    ax33.set_axis_off()

    plt.savefig('example_resulrt.png', format='png', bbox_inches='tight', transparent=True, dpi=300)