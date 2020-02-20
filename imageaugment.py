import augmentations
import cv2
import time
import argparse
import os

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

images = []

parser = argparse.ArgumentParser(description='Image Augmenter to generate extra data in image classification')

parser.add_argument('--d', help='path to directory for augmentation', action='store', default="")
parser.add_argument('--t', help='type of images to search for (required if using --d)', action='store')

parser.add_argument('--flip', help='horizontal flip', action='store_true')
parser.add_argument('--rotate', help='random rotation', action='store_true')
parser.add_argument('--shift', help='random shift', action='store_true')
parser.add_argument('--scale', help='random_scale', action='store_true')
parser.add_argument('--shear', help='random shear', action='store_true')
parser.add_argument('--affine', help='random affine transform', action='store_true')
parser.add_argument('--temp', help='random temperature', action='store_true')
parser.add_argument('--contrast', help='random contrast', action='store_true')
parser.add_argument('--noise', help='random salt and pepper noise', action='store_true')
parser.add_argument('--saturation', help='random saturation', action='store_true')
parser.add_argument('--random', help='default option, will randomly augment', action='store_true', default=True)

args = parser.parse_args()

path = os.path.abspath(args.d)
dirs = os.listdir(path)

for f in dirs:
    filename, file_extension = os.path.splitext(f)
    if file_extension == args.t:
        images.append(os.path.join(path, f))
        
if args.random:
    for image in images:
        aug = augmentations.Augmenter(image)
        img = aug.random_augment()
        show(img)