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
    mod_dim = 45

    min_label_num = 4
    max_label_num = 256


class SEblock(nn.Module):
    def __init__(self, channel, r=0.5):
        super(SEblock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        weight = self.fc(branch)

        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        scale = weight * x
        return scale


class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if not mid_channel:
            mid_channel = out_channel

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=[64, 128, 256, 512, 1024], classes=64, bilinear=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.classes = classes
        self.bilinear = bilinear
        if not out_channel:
            out_channel = [64, 128, 256, 512, 1024]

        self.inc = DoubleConv(in_channel, out_channel[0])
        self.se1 = SEblock(64)
        self.down1 = Down(out_channel[0], out_channel[1])
        self.se2 = SEblock(128)
        self.down2 = Down(out_channel[1], out_channel[2])
        self.se3 = SEblock(256)
        self.down3 = Down(out_channel[2], out_channel[3])
        factor = 2 if bilinear else 1

        self.up2 = Up(out_channel[3], out_channel[2] // factor, bilinear)
        self.up3 = Up(out_channel[2], out_channel[1] // factor, bilinear)
        self.up4 = Up(out_channel[1], out_channel[0] // factor, bilinear)

        self.out_conv = OutConv(out_channel[0], classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_se = self.se1(x1)
        x2 = self.down1(x1)
        x2_se = self.se2(x2)
        x3 = self.down2(x2)
        x3_se = self.se3(x3)
        x4 = self.down3(x3)
        x = self.up2(x4, x3_se)
        x = self.up3(x, x2_se)
        output = self.up4(x, x1_se)

        return output


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

    model = UNet(bilinear=False).to(device)
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
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite("seg_img/SE-UNet/seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)


if __name__ == '__main__':
    run()
