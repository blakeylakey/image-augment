# Image Augmenter

Small library to perform basic image manipulations with the intent of producing extra images in a dataset. Image augmenter can perform image flips, rotations, shifts, scales, shears, affine transformations, temperature changes, contrast changes, salt and pepper noise, saturation changes, or a completely random transformation of the previously mentioned augmentations. 

Ideally, you can use this image augmenter to supplement images in a coarse dataset for machine learning projects. 

## Usage

```python
from augmentations import Augmenter

img_path = 'path/to/image.jpg'
aug = Augmenter(img_path, affine=True ##, other options)

aug_img = aug.augment()
random_aug_img = aug.random_augment()
```

## License
[MIT](./LICENSE)