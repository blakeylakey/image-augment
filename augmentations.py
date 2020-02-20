import skimage as sk
import random
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

temperatures = {
    "1000": [255, 56, 0],
    "1100": [255, 71, 0],
    "1200": [255, 83, 0],
    "1300": [255, 93, 0],
    "1400": [255, 101, 0],
    "1500": [255, 109, 0],
    "1600": [255, 115, 0],
    "1700": [255, 121, 0],
    "1800": [255, 126, 0],
    "1900": [255, 131, 0],
    "2000": [255, 138, 18],
    "2100": [255, 142, 33],
    "2200": [255, 147, 44],
    "2300": [255, 152, 54],
    "2400": [255, 157, 63],
    "2500": [255, 161, 72],
    "2600": [255, 165, 79],
    "2700": [255, 169, 87],
    "2800": [255, 173, 94],
    "2900": [255, 177, 101],
    "3000": [255, 180, 107],
    "3100": [255, 184, 114],
    "3200": [255, 187, 120],
    "3300": [255, 190, 126],
    "3400": [255, 193, 132],
    "3500": [255, 196, 137],
    "3600": [255, 199, 143],
    "3700": [255, 201, 148],
    "3800": [255, 204, 153],
    "3900": [255, 206, 159],
    "4000": [255, 209, 163],
    "4100": [255, 211, 168],
    "4200": [255, 213, 173],
    "4300": [255, 215, 177],
    "4400": [255, 217, 182],
    "4500": [255, 219, 186],
    "4600": [255, 221, 190],
    "4700": [255, 223, 194],
    "4800": [255, 225, 198],
    "4900": [255, 227, 202],
    "5000": [255, 228, 206],
    "5100": [255, 230, 210],
    "5200": [255, 232, 213],
    "5300": [255, 233, 217],
    "5400": [255, 235, 220],
    "5500": [255, 236, 224],
    "5600": [255, 238, 227],
    "5700": [255, 239, 230],
    "5800": [255, 240, 233],
    "5900": [255, 242, 236],
    "6000": [255, 243, 239],
    "6100": [255, 244, 242],
    "6200": [255, 245, 245],
    "6300": [255, 246, 247],
    "6400": [255, 248, 251],
    "6500": [255, 249, 253],
    "6600": [254, 249, 255],
    "6700": [252, 247, 255],
    "6800": [249, 246, 255],
    "6900": [247, 245, 255],
    "7000": [245, 243, 255],
    "7100": [243, 242, 255],
    "7200": [240, 241, 255],
    "7300": [239, 240, 255],
    "7400": [237, 239, 255],
    "7500": [235, 238, 255],
    "7600": [233, 237, 255],
    "7700": [231, 236, 255],
    "7800": [230, 235, 255],
    "7900": [228, 234, 255],
    "8000": [227, 233, 255],
    "8100": [225, 232, 255],
    "8200": [224, 231, 255],
    "8300": [222, 230, 255],
    "8400": [221, 230, 255],
    "8500": [220, 229, 255],
    "8600": [218, 229, 255],
    "8700": [217, 227, 255],
    "8800": [216, 227, 255],
    "8900": [215, 226, 255],
    "9000": [214, 225, 255],
    "9100": [212, 225, 255],
    "9200": [211, 224, 255],
    "9300": [210, 223, 255],
    "9400": [209, 223, 255],
    "9500": [208, 222, 255],
    "9600": [207, 221, 255],
    "9700": [207, 221, 255],
    "9800": [206, 220, 255],
    "9900": [205, 220, 255],
    "10000": [207, 218, 255],
    "10100": [207, 218, 255],
    "10200": [206, 217, 255],
    "10300": [205, 217, 255],
    "10400": [204, 216, 255],
    "10500": [204, 216, 255],
    "10600": [203, 215, 255],
    "10700": [202, 215, 255],
    "10800": [202, 214, 255],
    "10900": [201, 214, 255],
    "11000": [200, 213, 255],
    "11100": [200, 213, 255],
    "11200": [199, 212, 255],
    "11300": [198, 212, 255],
    "11400": [198, 212, 255],
    "11500": [197, 211, 255],
    "11600": [197, 211, 255],
    "11700": [197, 210, 255],
    "11800": [196, 210, 255],
    "11900": [195, 210, 255],
    "12000": [195, 209, 255]
}

class Augmenter:
    def __init__(self, img_path, flip=False, rotate=False, shift=False, scale=False, shear=False, affine=False, temp=False, contrast=False, noise=False, saturation=False):
        self.img_path = img_path

        self.img = cv2.imread(self.img_path)

        self.options = {
            'flip': flip,
            'rotate': rotate,
            'shift': shift,
            'scale': scale,
            'shear': shear,
            'affine': affine,
            'temp': temp,
            'contrast': contrast,
            'noise': noise,
            'saturation': saturation
        }

    ######################################################################
    ## Augment using options
    def augment(self):
        image = self.img

        for option in self.options:
            if option == 'flip':
                image = self.horizontal_flip(self.img)
            if option == 'rotate':
                image = self.random_rotation(self.img)
            if option == 'shift':
                image = self.random_shift(self.img)
            if option == 'scale':
                image = self.random_scale(self.img)
            if option == 'shear':
                image = self.random_shear(self.img)
            if option == 'affine':
                image = self.random_affine(self.img)
            if option == 'temp':
                image = self.random_affine(self.img)
            if option == 'contrast':
                image = self.random_affine(self.img)
            if option == 'saturation':
                image = self.random_affine(self.img)
        
        return image

    ######################################################################
    ## Random Augmentations
    def random_augment(self):
        image = self.img

        for option in self.options:
            if random.random() >= .5:
                image = self.horizontal_flip(self.img)
            if random.random() >= .5:
                image = self.random_rotation(self.img)
            if random.random() >= .5:
                image = self.random_shift(self.img)
            if random.random() >= .5:
                image = self.random_scale(self.img)
            if random.random() >= .5:
                image = self.random_shear(self.img)
            if random.random() >= .5:
                image = self.random_affine(self.img)
            if random.random() >= .5:
                image = self.random_affine(self.img)
            if random.random() >= .5:
                image = self.random_affine(self.img)
            if random.random() >= .5:
                image = self.random_affine(self.img)
        
        return image
        

    ######################################################################
    ## Basic Image Transformations

    ## Horizontal Flips
    def horizontal_flip(self, img):
        return img[:,::-1]

    ## Random Rotation
    def random_rotation(self, img, min=-15, max=15):
        random_degree = random.uniform(min, max)
        return sk.transform.rotate(img, random_degree)

    ## Random Translation
    def random_shift(self, img, min=-15, max=15):
        vector = [random.uniform(min, max), random.uniform(min,max)]
        transform = sk.transform.AffineTransform(translation=vector)
        shifted = sk.transform.warp(img, transform, mode='constant', preserve_range=True)
        shifted = shifted.astype(img.dtype)
        return shifted

    ## Random Scaling
    def random_scale(self, img, min=0.85, max=1.15):
        vector = [random.uniform(min, max), random.uniform(min,max)]
        transform = sk.transform.AffineTransform(scale=vector)
        scaled = sk.transform.warp(img, transform, mode='constant', preserve_range=True)
        scaled = scaled.astype(img.dtype)
        return scaled

    ## Random Shearing
    def random_shear(self, img, min=-15, max=15):
        random_degree = 3.1415*(random.uniform(min, max)/180)
        transform = sk.transform.AffineTransform(shear=random_degree)
        sheared = sk.transform.warp(img, transform, mode='constant', preserve_range=True)
        sheared = sheared.astype(img.dtype)
        return sheared

    ## Random Affine Transformation
    ## Randomly chooses to apply scale/rotate/shift/shear/flip
    def random_affine(self, img, scale_max=0.85, scale_min=1.15, rotate_min=-15, rotate_max=15, shift_min=-15, shift_max=15, shear_min=-15, shear_max=15):
        effects = {
            'scale': round(random.random()),
            'rotate': round(random.random()),
            'shift': round(random.random()),
            'shear': round(random.random()),
            'flip': round(random.random())
        }

        if effects['scale']:
            scale_vector = [random.uniform(scale_min, scale_max), random.uniform(scale_min, scale_max)]
        else:
            scale_vector = None

        if effects['rotate']:
            rotate_degree = 3.1415*(random.uniform(rotate_min, rotate_max)/180)
        else:
            rotate_degree = None

        if effects['shift']:
            shift_vector = [random.uniform(shift_min, shift_max), random.uniform(shift_min, shift_max)]
        else:
            shift_vector = None
        
        if effects['shear']:
            shear_degree = 3.1415*(random.uniform(shear_min, shear_max)/180)
        else:
            shear_degree = None
        
        transform = sk.transform.AffineTransform(scale=scale_vector, rotation=rotate_degree, translation=shift_vector, shear=shear_degree)
        affined = sk.transform.warp(img, transform, mode='constant', preserve_range=True)
        affined = affined.astype(img.dtype)

        if effects['flip']:
            return affined[:,::-1]
        else:
            return affined

    ######################################################################
    ## More Fun Image Transformations

    ## Changes Temperature (K)
    def random_temperature(self, img):
        img_copy = np.copy(img)
        keys = list(temperatures.keys())
        k = temperatures[keys[int(random.random()*len(keys))]]
        r, g, b = k[0]/255.0, k[1]/255.0, k[2]/255.0

        for row in img_copy:
            for pixel in row:
                pixel[0] *= b
                pixel[1] *= g
                pixel[2] *= r

        return img_copy

    ## Changes Brightness & Contrast (alpha * pixel_value + beta)
    ## alpha = contrast, beta = brightness
    def random_bright_contrast(self, img, contrast_min=0.85, contrast_max=1.15, beta_min=-50, beta_max=50):
        img_copy = np.copy(img)
        alpha = random.uniform(contrast_min, contrast_min)
        beta = random.uniform(beta_min, beta_max)

        for row in img_copy:
            for pixel in row:
                pixel[0] = max(min(alpha*pixel[0] + beta, 255.0),0.0)
                pixel[1] = max(min(alpha*pixel[1] + beta, 255.0),0.0)
                pixel[2] = max(min(alpha*pixel[2] + beta, 255.0),0.0)

        return img_copy

    ## Adds Salt & Pepper Noise
    def random_noise(self, img):
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = 0.02
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in img.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in img.shape]
        out[tuple(coords)] = 0
        return out

    ## Changes Saturation and Value
    def random_hsv(self, img, sat_min=0.85, sat_max=1.15, val_min=0.85, val_max=1.15):
        saturation = random.uniform(sat_min, sat_max)
        value = random.uniform(val_min, val_max)

        img_copy = np.copy(img)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

        for row in img_copy:
            for pixel in row:
                pixel[1] = max(min(saturation*pixel[1], 255.0), 0.0)
                pixel[2] = max(min(value*pixel[2], 255.0), 0.0)

        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_HSV2BGR)

        return img_copy