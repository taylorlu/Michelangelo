import tensorflow as tf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,H,W,C] weights for each pixel
    :return:
    """
    if loss_type == 'l1':
        mae = tf.keras.losses.MeanAbsoluteError()
        loss = mae(target, logit, sample_weight=weight)
        return loss
    elif loss_type == 'l2':
        mse = tf.keras.losses.MeanSquaredError()
        loss = mse(target, logit, sample_weight=weight)
        return loss
    else:
        raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


def composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
    """
    Alpha composition loss
    """
    merged = fg * alpha + bg * (1 - alpha)
    return regression_loss(merged, image, loss_type=loss_type, weight=weight)


def lap_loss(logit, target, weight):
    '''
    Based on FBA Matting implementation:
    https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
    '''
    gauss_filter = np.array([[[[1., 4., 6., 4., 1.],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]]]], dtype=np.float32)
    gauss_filter /= 256.
    gauss_filter = np.transpose(gauss_filter, [2, 3, 0, 1])

    def conv_gauss(x, gauss_filter):
        x = tf.pad(x, paddings=[[0,0], [2,2], [2,2], [0,0]], mode='REFLECT')
        x = tf.nn.conv2d(x, gauss_filter, strides=1, padding='VALID')
        return x

    def downsample(x):
        return x[:, :, ::2, ::2]

    def upsample(x):
        cc = tf.concat([x, tf.zeros_like(x)], 2)
        cc = tf.reshape(cc, [cc.shape[0], -1, x.shape[2], cc.shape[3]])
        cc = tf.transpose(cc, [0, 2, 1, 3])
        cc = tf.concat([cc, tf.zeros_like(cc)], 2)
        cc = tf.reshape(cc, [cc.shape[0], x.shape[1]*2, x.shape[2]*2, cc.shape[3]])
        x_up = tf.transpose(cc, [0, 2, 1, 3])
        return conv_gauss(x_up, 4*gauss_filter)

    def lap_pyramid(x, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current)
            down = downsample(filtered)
            up = upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def weight_pyramid(x, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            down = downsample(current)
            pyr.append(current)
            current = down
        return pyr

    pyr_logit = lap_pyramid(logit, max_levels=5)
    pyr_target = lap_pyramid(target, max_levels=5)
    pyr_weight = weight_pyramid(weight, max_levels=5)

    loss_sum = 0
    for i, A in enumerate(zip(pyr_logit, pyr_target, pyr_weight)):
        loss_sum += (regression_loss(A[0], A[1], weight=A[2]) * (2**i))

    return loss_sum


if(__name__=='__main__'):

    gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]]).cuda()
    gauss_filter /= 256.
    gauss_filter = gauss_filter.repeat(1, 1, 1, 1)

    input = cv2.imread(r'E:\CVDataset\PhotoMatte85\0051115Q_000001_0003.png')[612:1124, 612:1124, :]
    pred = input.transpose([2,0,1])[np.newaxis, 0:1, ...]
    pred = torch.from_numpy(pred).cuda()
    lap_loss(pred, gauss_filter)
