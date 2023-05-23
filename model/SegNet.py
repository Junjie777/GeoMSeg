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
    mod_dim1 = 128
    mod_dim2 = 64
    gpu_id = 0

    min_label_num = 4
    max_label_num = 256


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, dilation=1, is_batchnorm=True):
        super(conv2DBatchNormRelu, self).__init__()
        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.cbr_unit = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=bias, dilation=dilation),
                nn.ReLU(inplace=True)
            )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        # outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class segnet(nn.Module):
    def __init__(self, in_channels=3, mod_dim=64):
        super(segnet, self).__init__()
        self.down1 = segnetDown2(in_channels=in_channels, out_channels=64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, mod_dim)

        self.finconv = conv2DBatchNormRelu(64, mod_dim, 3, 1, 1)

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)

        up2 = self.up2(down2, indices=indices_2, output_shape=unpool_shape2)
        outputs = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        return outputs


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

    model = segnet().to(device)
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
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
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

    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite("seg_img/SegNet/seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)


if __name__ == '__main__':
    run()
