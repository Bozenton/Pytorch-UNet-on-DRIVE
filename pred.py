import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.DriveDataset import DriveDataset
from unet import UNet
from utils.utils import plot_img_and_mask

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
    in_file = './datasets/test/images/01_test.tif'
    out_dir = Path('./datasets/test/predict')
    weight_file = './checkpoints/checkpoint_epoch2.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.load_state_dict(torch.load(weight_file, map_location=device))

    img = Image.open(in_file)
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=1,
                       out_threshold=0.5,
                       device=device)
    mask = mask_to_image(mask)
    mask.save(os.path.join(out_dir, '01_test_pred.gif'))
