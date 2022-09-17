import os
import time

import cv2.cv2 as cv2
import numpy as np
from skimage import segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Args(object):
    input_image_path = 'image/map03.png'  # image/coral.jpg image/tiger.jpg
    train_epoch = 2 ** 6
    mod_dim1 = 64
    mod_dim2 = 45
    gpu_id = 0

    min_label_num = 4  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def ConvBNAct(in_channels,out_channels,kernel_size=3, stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups),
            nn.BatchNorm2d(out_channels),
            Swish()
        )


def Conv1x1BNAct(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        Swish(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid_channels = channels // ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expansion_factor = expansion_factor
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNAct(in_channels, mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels),
            SEBlock(mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out + self.shortcut(x)) if self.stride==1 else out
        return out


class EfficientNet(nn.Module):
    params = {
        'efficientnet_b0': (1.0, 1.0, 224, 0.4),
        'efficientnet_b1': (1.8, 1.1, 240, 0.2),
        'efficientnet_b2': (1.1, 1.4, 260, 0.3),
        'efficientnet_b3': (1.2, 1.4, 300, 0.2),
        'efficientnet_b4': (1.6, 1.8, 380, 0.3),
        'efficientnet_b5': (1.7, 2.1, 456, 0.4),
        'efficientnet_b6': (1.6, 2.6, 528, 0.6),
        'efficientnet_b7': (1.5, 3.1, 600, 0.5),
    }
    def __init__(self, subtype='efficientnet_b0', num_classes=64):
        super(EfficientNet, self).__init__()
        self.width_coeff = self.params[subtype][0]
        self.depth_coeff = self.params[subtype][1]
        self.dropout_rate = self.params[subtype][3]
        self.depth_div = 8

        self.stage1 = ConvBNAct(3, self._calculate_width(32), kernel_size=3, stride=1)
        self.stage2 = self.make_layer(self._calculate_width(32), self._calculate_width(64), kernel_size=3, stride=1, block=self._calculate_depth(1))
        self.stage3 = self.make_layer(self._calculate_width(64), self._calculate_width(128), kernel_size=3, stride=1, block=self._calculate_depth(2))
        self.stage4 = self.make_layer(self._calculate_width(128), self._calculate_width(86), kernel_size=3, stride=1, block=self._calculate_depth(2))
        self.stage5 = self.make_layer(self._calculate_width(86), self._calculate_width(64), kernel_size=3, stride=1, block=self._calculate_depth(3))
        self.stage6 = self.make_layer(self._calculate_width(80), self._calculate_width(112), kernel_size=5, stride=1, block=self._calculate_depth(3))
        self.stage7 = self.make_layer(self._calculate_width(112), self._calculate_width(192), kernel_size=5, stride=2, block=self._calculate_depth(4))
        self.stage8 = self.make_layer(self._calculate_width(192), self._calculate_width(64), kernel_size=3, stride=1, block=self._calculate_depth(1))

        # self.classifier = nn.Sequential(
        #     Conv1x1BNAct(320, 1280),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Dropout2d(0.2),
        #     Flatten(),
        #     nn.Linear(1280, num_classes)
        # )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def _calculate_width(self, x):
        x *= self.width_coeff
        new_x = max(self.depth_div, int(x + self.depth_div / 2) // self.depth_div * self.depth_div)
        if new_x < 0.9 * x:
            new_x += self.depth_div
        return int(new_x)

    def _calculate_depth(self, x):
        return int(math.ceil(x * self.depth_coeff))

    def make_layer(self, in_places, places, kernel_size, stride, block):
        layers = []
        layers.append(MBConvBlock(in_places, places, kernel_size, stride))
        for i in range(1, block):
            layers.append(MBConvBlock(places, places, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # x = self.stage6(x)
        # x = self.stage7(x)
        # x = self.stage8(x)
        # out = self.classifier(x)
        return x
class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        # 仿照FCN网络结构进行改写
        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, mod_dim2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


def run():
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    image = cv2.imread(args.input_image_path)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel)
    image = cv2.erode(image, kernel)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=40, sigma=0.4, min_size=60)
    print('seg_map.shape', seg_map.shape)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]
    print('seg_lab.len', len(seg_lab))

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print('image shape', image.shape)
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    print('tensor shape', tensor.shape)
    tensor = torch.from_numpy(tensor).to(device)

    model = EfficientNet('efficientnet_b0').to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = image.reshape((-1, 3))
    print('image_flatten', image_flatten.shape)
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image
    print('show_image', show.shape)

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        print(output.shape)
        output = output.permute(1, 2, 0).view(-1, 64)
        target = torch.argmax(output, 1)
        print('target.shape', target.shape)
        im_target = target.data.cpu().numpy()

        '''refine'''
        #j = 1
        for index in seg_lab:
            u_labels, hist = np.unique(im_target[index], return_counts=True)
            im_target[index] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
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
    cv2.imwrite("seg_img/seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)


if __name__ == '__main__':
    run()