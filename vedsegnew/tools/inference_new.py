import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vedaseg.runners import InferenceRunner
from vedaseg.utils import Config
import  os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

CLASSES = ('background', 'Crack', 'Ketou', 'EdgeCollapse', 'Bubble', 'Blowhole',
           'ToolMark', 'Other', 'YaHen', 'WhiteSpot', 'YellowSpot', 'VeryDark', 'VeryLight',
           'Dark', 'Light')

# CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor')

# PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0],  [0, 0, 128],
#            [128, 0, 128], [0, 128, 128],  [64, 0, 0],
#            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
#            [192, 0, 128], [64, 128, 128],  [0, 64, 0],
#            [128, 64, 0]]

PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
                   [255, 0, 255], [0, 128, 0], [255, 140, 0], [0, 255, 255],
                   [255, 192, 203], [154, 205, 50], [0, 255, 0], [128, 128, 0], [0, 0, 128],
                   [128, 0, 128], [128, 128, 128]]
#
# np.asarray([[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
#                    [255, 0, 255], [0, 128, 0], [255, 140, 0], [0, 255, 255],
#                    [255, 192, 203], [154, 205, 50], [0, 255, 0], [128, 128, 0], [0, 0, 128],
#                    [128, 0, 128], [128, 128, 128]])


def inverse_pad(output, image_shape):
    h, w = image_shape
    return output[:h, :w]


def plot_result(img, mask, cover):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Vedaseg Demo", y=0.95, fontsize=16)

    ax[0].set_title('image')
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[1].set_title('mask')
    ax[1].imshow(mask)

    ax[2].set_title('cover')
    ax[2].imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    plt.show()


def result(fname,
           pred_mask,
           classes,
           multi_label=False,
           palette=None,
           show=False,
           out=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(len(classes), 3))
    else:
        palette = np.array(palette)
    img_ori = cv2.imread(fname)
    mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        if multi_label:
            mask[pred_mask[:, :, label] == 1] = color
        else:
            mask[pred_mask == label, :] = color

    cover = img_ori * 0.5 + mask * 0.5
    cover = cover.astype(np.uint8)

    if out is not None:
        _, fullname = os.path.split(fname)
        fname, _ = os.path.splitext(fullname)
        save_dir = os.path.join(out, fname)
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, fullname.replace('.bmp', '_img.png')), img_ori)
        cv2.imwrite(os.path.join(out, fullname.replace('.bmp', '_mask.png')), mask)
        cv2.imwrite(os.path.join(out, fullname.replace('.bmp', '_cover.png')), cover)
        # cv2.imwrite(os.path.join(save_dir, 'img.png'), img_ori)
        # cv2.imwrite(os.path.join(save_dir, 'mask.png'), mask)
        # cv2.imwrite(os.path.join(save_dir, 'cover.png'), cover)
        if multi_label:
            for i in range(pred_mask.shape[-1]):
                cv2.imwrite(os.path.join(out, classes[i] + '.png'),
                            pred_mask[:, :, i] * 255)

    if show:
        plot_result(img_ori, mask, cover)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference a segmentatation model')
    parser.add_argument('--config', type=str,default='/media/anji/sp4/datasets/modelsv/shushu/1228/coco_dpv3plusres50.py',
                        help='config file path')
    parser.add_argument('--checkpoint', type=str,default='/media/anji/sp4/datasets/modelsv/shushu/1228/big_shushu.pth',
                        help='checkpoint file path')
    parser.add_argument('--image', type=str,default='/media/anji/sp4/datasets/jinlong/newsty/traindata/valbmp',
                        help='input image path')
    parser.add_argument('--show', action='store_true',
                        help='show result images on screen')
    parser.add_argument('--out', default='/media/anji/sp4/datasets/jinlong/newsty/traindata/valbmp_segnew',
                        help='folder to store result images')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    multi_label = cfg.get('multi_label', False)
    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    for imgsg in os.listdir(args.image):
        sgpath = os.path.join(args.image,imgsg)
        image = cv2.imread(sgpath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]
        dummy_mask = np.zeros(image_shape)

        output = runner(image, [dummy_mask])
        output1 = output
        classlist = output[(output1 > 0)].tolist()
        # tensor.numpy().tolist()
        print('calss==', output1[(output1 > 0)])
        if len(classlist) > 0:
            maxclsnm = max(classlist, key=classlist.count)
            print('###maxclass####=', max(classlist, key=classlist.count))
        else:
            maxclsnm = None
        if multi_label:
            output = output.transpose((1, 2, 0))

        output = inverse_pad(output, image_shape)

        result(sgpath, output, multi_label=multi_label,
               classes=CLASSES, palette=PALETTE, show=args.show,
               out=args.out)


if __name__ == '__main__':
    main()
