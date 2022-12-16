from math import floor
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

# rgb order
PIXELS = (0.114, 0.587, 0.299)
# bgr order
# PIXELS = (0.299, 0.587, 0.114)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def flatten_conv_feature(feature):
    feature_size = feature.size()[1:]
    result = reduce(lambda a, b: a*b, feature_size)
    return result


def preprocess_state(state):
    img = np.squeeze(state.copy())
    img = np.dot(img[..., :3], PIXELS)
    h, w = img.shape
    img = np.reshape(img, (1, 1, h, w))
    return img


if __name__ == '__main__':
    # h, w = conv_output_shape(h_w=(84, 84), kernel_size=8, stride=4, pad=1)
    # print('conv1 output shape', h, w)
    # h, w = conv_output_shape(h_w=(h, w), kernel_size=4, stride=2, pad=1)
    # print('conv2 output shape', h, w)
    # h, w = conv_output_shape(h_w=(h, w), kernel_size=3, stride=1, pad=1)
    # print('conv3 output shape', h, w)

    # dim_length = flat_conv_feature((1, 3, 3, 3))
    # print(dim_length)

    preprocess_state(None)
