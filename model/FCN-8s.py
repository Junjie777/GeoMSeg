from typing import Dict
import os
import time

import cv2.cv2 as cv2
import numpy as np
from skimage import segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class Args(object):
    input_image_path = ''
    train_epoch = 2 ** 6
    gpu_id = 0

    min_label_num = 4
    max_label_num = 256


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(SingleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            # DoubleConv(in_channels, out_channels)
            SingleConv(in_channels, out_channels)
        )


class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=64, bilinear=True, base_c=64):
        super(FCN, self).__init__()

        self.in_conv = SingleConv(in_channels, base_c // 2)
        self.down1 = Down(base_c // 2, base_c)
        self.down2 = Down(base_c, base_c * 2)
        self.down3 = Down(base_c * 2, base_c * 4)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 4, base_c * 8)
        self.down5 = Down(base_c * 8, base_c * 8)

        self.in_conv1 = SingleConv(base_c * 4, base_c * 2)
        self.in_conv2 = SingleConv(base_c * 2, num_classes)

        self.upsample2x = nn.ConvTranspose2d(in_channels=base_c * 8, out_channels=base_c * 8, kernel_size=2, stride=2)
        self.upsample8x = nn.ConvTranspose2d(in_channels=base_c * 4, out_channels=base_c * 4, kernel_size=8, stride=8)
        self.upsample16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upsample32x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x6 = self.upsample2x(x5)
        x6 += x4
        x6 = self.upsample2x(x6)
        x6 = self.conv1(x6)
        x6 += x3
        x6 = self.upsample8x(x6)

        x7 = self.in_conv1(x6)
        out = self.in_conv2(x7)

        return out


def run():
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    image = cv2.imread(args.input_image_path)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel)
    image = cv2.erode(image, kernel)

    seg_map = segmentation.felzenszwalb(image, scale=40, sigma=0.4, min_size=60)
    print('seg_map.shape', seg_map.shape)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]
    print('seg_lab.len', len(seg_lab))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print('image shape', image.shape)
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0

    tensor = tensor[np.newaxis, :, :, :]
    print('tensor shape', tensor.shape)
    tensor = torch.from_numpy(tensor).to(device)

    model = FCN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = image.reshape((-1, 3))
    print('image_flatten', image_flatten.shape)
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image
    print('show_image', show.shape)

    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        optimizer.zero_grad()
        output = model(tensor)[0]
        print(output.shape)
        output = output.permute(1, 2, 0).view(-1, 64)
        target = torch.argmax(output, 1)
        print('target.shape', target.shape)
        im_target = target.data.cpu().numpy()

        for index in seg_lab:
            u_labels, hist = np.unique(im_target[index], return_counts=True)
            im_target[index] = u_labels[np.argmax(hist)]

        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int32) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        cv2.imshow("seg_pt", show)
        cv2.waitKey(1)

        print('Loss:', batch_idx, loss.item())
        length = len(un_label)
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite("seg_img/FCN8/seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)


if __name__ == '__main__':
    run()
