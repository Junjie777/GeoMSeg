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
    mod_dim = 64
    gpu_id = 0

    min_label_num = 4
    max_label_num = 256


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, mod_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


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
    print('seg_map.shape1', seg_map.shape)
    seg_map = seg_map.flatten()
    print('seg_map.shape2', seg_map.shape)
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

    model = MyNet(inp_dim=3, mod_dim=args.mod_dim).to(device)
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
        print('output.shape1', output.shape)
        output = output.permute(1, 2, 0).view(-1, args.mod_dim)
        print('output.shape2', output.shape)
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
    cv2.imwrite("seg_img/CNN/seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)


if __name__ == '__main__':
    run()