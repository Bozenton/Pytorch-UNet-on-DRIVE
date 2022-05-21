import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor

from utils.DriveDataset import DriveDataset
from utils.metrics import get_accuracy, get_specificity, get_sensitivity
from unet import UNet

from tqdm import tqdm

def predict_img(net: UNet,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(DriveDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    # 注意，这里的mask不是指分割的区域，而是ROI
    img_dir = Path('./datasets/test/images')
    mask_dir = Path('./datasets/test/mask')
    gt_dir = Path('./datasets/test/manual')  # ground truth
    out_dir = Path('./datasets/test/predict')
    assert os.path.exists(img_dir), "{} does not exist".format(img_dir)
    assert os.path.exists(mask_dir), "{} does not exist".format(mask_dir)
    assert os.path.exists(gt_dir), "{} does not exist".format(gt_dir)
    assert os.path.exists(out_dir), "{} does not exist".format(out_dir)
    test_dataset = DriveDataset(img_dir, mask_dir, scale=1, mask_suffix='_test_mask')
    test_loader = DataLoader(test_dataset, shuffle=False)

    weight_file = './checkpoints/checkpoint_epoch20.pth'
    assert os.path.exists(weight_file), "Weight file {} does not exist, you can download it from ...".format(
        weight_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.load_state_dict(torch.load(weight_file, map_location=device))

    accuarcy = []
    sensitivity = []
    specificity = []

    for batch in tqdm(test_loader, total=len(test_loader), desc='Test round', unit='batch', leave=False):
        image_name = batch['name'][0]   # batch 0
        image = batch['image']  # torch.Size([1, 3, 584, 565]) BCHW
        mask = batch['mask']  # torch.Size([1, 584, 565]) BCHW
        image = image.to(device=device)

        # load ground truth
        image_id = image_name.split('_')[0]  # str
        gt_file = os.path.join(gt_dir, image_id+'_manual1.gif')
        assert os.path.exists(gt_file), 'grond truth {} does not exist'.format(gt_file)
        gt = ToTensor()(Image.open(gt_file))
        gt = gt[0, ...]

        with torch.no_grad():
            output = net(image)  # torch.Size([1, 2, 584, 565]) BCHW
            pred = F.one_hot(output.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            pred = pred[0, 1, ...]
            assert gt.shape == pred.shape, "the shape of ground true {} and predict {} should be the same".format(gt.shape, pred.shape)

            acc = get_accuracy(pred, gt)
            sen = get_sensitivity(pred, gt)
            spc = get_specificity(pred, gt)
            accuarcy.append(acc)
            sensitivity.append(sen)
            specificity.append(spc)

            pred_ndarray = pred.numpy()
            pred_pil = mask_to_image(pred_ndarray)
            pred_pil.save(os.path.join(out_dir, image_name+'_pred.gif'))
    print(accuarcy)
    print(sensitivity)
    print(specificity)


