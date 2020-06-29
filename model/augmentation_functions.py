import tensorflow as tf
import numpy as np
import cv2
from tensorflow_core.python.ops.gen_image_ops import sample_distorted_bounding_box_v2


def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
        lambda: func(x), lambda: x)


def color_jitter(x, s=1.0):
    # You can also shuffle the order of following augmentations each time they are applied.
    x = tf.image.random_brightness(x, max_delta=0.8 * s)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_saturation(x, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
    x = tf.image.random_hue(x, max_delta=0.2 * s)
    x = tf.clip_by_value(x, 0, 1)
    return x


def color_drop(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


def color_distortion(image, s=1.0):
    # Image is a tensor with value range in [0, 1]. "s" is the strength of color distortion.

    # Randomly apply transformation with probability p:
    image = random_apply(color_jitter, x=image, p=0.8)
    image = random_apply(color_drop, x=image, p=0.2)
    return image


def crop_and_resize(image, height, width, minCrop=0.08):
    """Make a random crop and resize it to height `height` and width `width`.

  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.

  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])     #Also ist bbox ein Viereck mit (ymin,xmin,ymax,xmax)=(0,0,1,1)
    aspect_ratio = width / height

    begin, size, _ = sample_distorted_bounding_box_v2(
        image_size=tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,     #Mindestens N% des Originalbilds müssen sich in der cropped version wiederfinden lassen
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),     #cropped area hat shape im Bereich [0.75*width/height, 1.333*width/height]
        area_range=(minCrop,1),    #cropped area muss in diesem Bereich des Originalbilds liegen (Werte von SimCLR-Github, obwohl nur >0.1 als untere Grenze Sinn macht?)
        max_attempts=100    #Nach 100 tries einfach das Originalbild beibehalten
    )
    slice_of_image=tf.slice(image, begin, size)
    return tf.image.resize(slice_of_image, size=(width,height), method='bicubic')

    # image = distorted_bounding_box_crop(
    #     image,
    #     bbox,
    #     min_object_covered=0.1,
    #     aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
    #     area_range=(0.08, 1.0),
    #     max_attempts=100,
    #     scope=None)
    # return tf.compat.v1.image.resize_bicubic([image], [height, width])[0]


def random_crop_with_resize(image, height, width, p=1.0, minCrop=0.08):
    """Randomly crop and resize an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: Probability of applying this transformation.
    minCrop: Minimal possible size of the crop. E.g. crop_height = [minCrop, 1] * height.
    At least 10% of original image can automatically be found in the crop though.

  Returns:
    A preprocessed image `Tensor`.
  """

    def _transform(image):
        image = crop_and_resize(image, height, width, minCrop=minCrop)
        return image

    return random_apply(_transform, p=p, x=image)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # 50% chance to blur image
        prob = np.random.random_sample()    # prob = [0,1)

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min    #(max-min)*0+min ... (max-min)*1+min -> Intervall: sigma = [min, max)
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)  #Verwendete Argumente: src, ksize, sigmaX
            # src=sample:   input image
            # ksize:        Gaussian Kernel Size [height, width]. height and width SHOULD BE ODD and can have different values.
            # sigmaX=sigma: Kernel standard deviation along X-axis (horizontal direction).
            # sigmaY:       If only sigmaX is specified, sigmaY is taken as equal to sigmaX. (so this happens if simgaY=None (which is the default))
        return sample


def gaussian_filter(img1, img2):    #images with same "image.shape[1]"
    k_size = int(img1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width AND SHOULD BE ODD (see func "cv2.GaussianBlur")
    gaussian_ope = GaussianBlur(kernel_size=k_size, min=0.1, max=2.0)
    [img1, ] = tf.py_function(gaussian_ope, [img1], [tf.float32])   #tf.py_function(function_to_use, input for function_to_use,
    [img2, ] = tf.py_function(gaussian_ope, [img2], [tf.float32])   #               tensorflow_datatype indicating what function_to_use returns)
    return img1, img2