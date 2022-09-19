import os
import time
import cv2.cv2 as cv2
import numpy as np
from skimage import segmentation
import torch
from torch import nn
from torch.nn import functional as F
import extractors
import warnings
warnings.filterwarnings("ignore")


class Args(object):
    input_image_path = 'image/map03.png'
    train_epoch = 2 ** 6
    mod_dim1 = 64
    mod_dim2 = 45
    gpu_id = 0

    min_label_num = 4
    max_label_num = 256

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        print(feats.size())
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, mod_dim=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=256, backend='resnet34',
                 pretrained=False):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 128)
        self.up_3 = PSPUpsample(128, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, mod_dim, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, mod_dim)
        )

    def forward(self, x):
        print('x:', x.size())
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p);print('p:', p.size())

        p = self.up_2(p);print('p:', p.size())
        p = self.drop_2(p);print('p:', p.size())

        p = self.up_3(p);print('p:', p.size())
        p = self.drop_2(p);print('p:', p.size())

        res = self.final(p);print('res1:', res.size())

        return res

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

    # '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=40, sigma=0.4, min_size=60)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
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

    model = PSPNet(psp_size=512, mod_dim=45, deep_features_size=256).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        print('target.shape', target.shape)
        im_target = target.data.cpu().numpy()

        '''refine'''
        #j = 1
        for index in seg_lab:
            u_labels, hist = np.unique(im_target[index], return_counts=True)
            im_target[index] = u_labels[np.argmax(hist)]

        '''backward'''r
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