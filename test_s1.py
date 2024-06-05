import torch
import torch.nn.functional as F
import gc

import numpy as np
import os, argparse

import cv2
from scipy import misc
import imageio
from PIL import Image
from tqdm import tqdm
from model.ProFact import ProFact

def parse_args():
    parser = argparse.ArgumentParser(
        description='test (and eval) a model')
    parser.add_argument("--ver", type=str, default="b3", help="base model")
    parser.add_argument('--testsize', type=int, default=1024, help='testing size')
    parser.add_argument('--dataset_path', type=str, default='./samples/', help='test config file path')
    parser.add_argument('--model_path', type=str, default='./checkpoint_save/profact_casia2.pth', help='test config model path')  ##
    parser.add_argument('--save_path', type=str, default='./results/', help='save path for results')  ##
    args = parser.parse_args()

    return args

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TestDataset:
    def __init__(self, test_path, testsize):
        self.testsize = testsize
        self.test_path = test_path
        self.filelist = sorted(os.listdir(self.test_path))
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.d_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.filelist)
        self.index = 0

    def load_data(self):
        image, img_size = self.rgb_loader(self.test_path + self.filelist[self.index])
        if img_size[0] < 512 or img_size[1] < 512:
            image = self.transform(image).unsqueeze(0)
        else:
            image = self.d_transform(image).unsqueeze(0)
        name = self.filelist[self.index].split('/')[-1]
        if not name.endswith('.png'):
            name = name.split(".")[0] + '.png'
        self.index += 1
        return image, name, img_size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img_size = img.size
            return img.convert('RGB'), img_size


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = np.logical_and(premask, groundtruth).sum().astype(np.float64)
    false_pos = np.logical_and(premask, gt_inv).sum().astype(np.float64)
    false_neg = np.logical_and(seg_inv, groundtruth).sum().astype(np.float64)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


def main():
    args = parse_args() 
    dataset_path = args.dataset_path
    model = ProFact(ver = args.ver, pretrained = False)  ##
    print(args.model_path)
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])

    model.cuda()
    model.eval()

    from utils.metrics import decompose, rm_and_make_dir, merge
    import shutil
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + 'tamper/'
    test_size = str(args.testsize)
    decompose(image_root, test_size)
    print('Decomposition complete.')

    test_loader = TestDataset(test_path='test_out/temp/input_decompose_' + test_size + '/', testsize=int(test_size))
    path_out = 'test_out/temp/input_decompose_' + test_size + '_pred/'
    rm_and_make_dir(path_out)
    for i in tqdm(range(test_loader.size)):
        image, name, img_size = test_loader.load_data()
        image = image.cuda()
        _, pred = model(image) ##

        if img_size[0] < 512 or img_size[1] < 512:
            pred = F.upsample(pred, size=(img_size[1], img_size[0]), mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy().squeeze()
        pred = ((pred - pred.min()) / (pred.max() - pred.min() + 1e-8) * 255).astype('uint8') 
        pred = Image.fromarray(pred)
        pred.save(path_out + name)
    print('Prediction complete.')

    if os.path.exists('test_out/temp/input_decompose_' + test_size + '/'):
        shutil.rmtree('test_out/temp/input_decompose_' + test_size + '/')
    save_path = merge(image_root, path_out, save_path, test_size)
    print('Merging complete.')

    path_gt = args.dataset_path +'gt/'  ##

    if os.path.exists(path_gt):
        flist = sorted(os.listdir(save_path))  ##
        f1, iou = [], []
        for file in tqdm(flist):
            pre = cv2.imread(save_path + file)  ##

            file_gt = path_gt + file[:-4] + '.png'   
            gt = cv2.imread(file_gt)

            H, W, C = pre.shape
            Hg, Wg, C = gt.shape
            if H != Hg or W != Wg:
                gt = cv2.resize(gt, (W, H))
                gt[gt > 127] = 255
                gt[gt <= 127] = 0

            pre[pre > 127] = 255
            pre[pre <= 127] = 0
            a, b = metric(pre / 255, gt / 255)

            f1.append(a)
            iou.append(b)

        print('Evaluation: F1: %5.4f, IOU: %5.4f' % (np.mean(f1), np.mean(iou)))


if __name__ == '__main__':
    main()
