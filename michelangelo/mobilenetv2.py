import tensorflow as tf
import numpy as np
from functools import reduce
import traceback

from tensorflow.python.ops.gen_math_ops import mod

def dynamic_memory_allocation():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except Exception:
            traceback.print_exc()

dynamic_memory_allocation()


class InvertedResidual(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, stride, expansion):
        super(InvertedResidual, self).__init__()
        hidden_dim = filterIn * expansion
        self.conv = tf.keras.Sequential()
        self.use_res_connect = stride == 1 and filterIn == filterOut

        if(expansion==1):
            self.conv.add(tf.keras.layers.Conv2DTranspose(hidden_dim, 3, (stride, stride), 'same', groups=hidden_dim, use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU(max_value=6))
            self.conv.add(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
        else:
            self.conv.add(tf.keras.layers.Conv2D(hidden_dim, 1, (1, 1), 'valid', use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU(max_value=6))
            self.conv.add(tf.keras.layers.Conv2DTranspose(hidden_dim, 3, (stride, stride), 'same', groups=hidden_dim, use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU(max_value=6))
            self.conv.add(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())

    def call(self, x):
        if(self.use_res_connect):
            return x + self.conv(x)
        else:
            return self.conv(x)


class Residual(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, stride, expansion):
        super(Residual, self).__init__()
        hidden_dim = filterIn * expansion
        self.conv = tf.keras.Sequential()
        self.use_res_connect = stride == 1 and filterIn == filterOut

        if(expansion==1):
            self.conv.add(tf.keras.layers.Conv2D(hidden_dim, 3, (stride, stride), 'same', groups=hidden_dim, use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU(max_value=6))
            self.conv.add(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
        else:
            self.conv.add(tf.keras.layers.Conv2D(hidden_dim, 1, (1, 1), 'valid', use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU(max_value=6))
            self.conv.add(tf.keras.layers.Conv2D(hidden_dim, 3, (stride, stride), 'same', groups=hidden_dim, use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU(max_value=6))
            self.conv.add(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.conv.add(tf.keras.layers.BatchNormalization())

    def call(self, x):
        if(self.use_res_connect):
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(tf.keras.models.Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        expansion = 6
        filter = 32
        last_filter = 512
        interverted_residual_setting = [
            # t, c, n, s
            [1        , 16, 1, 1],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]
        self.enc_filters = [16, 24, 32, 96, 512]

        # building first layer
        self.features = tf.keras.Sequential([self.conv_bn(filter, 2)])

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            for i in range(n):
                if i == 0:
                    self.features.add(Residual(filter, c, s, expansion=t))
                else:
                    self.features.add(Residual(filter, c, 1, expansion=t))
                filter = c

        # building last several layers
        self.features.add(self.conv_1x1_bn(last_filter))

    def conv_bn(self, filter, stride):
        return tf.keras.Sequential([tf.keras.layers.Conv2D(filter, 3, (stride, stride), 'same', use_bias=False),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(max_value=6)])

    def conv_1x1_bn(self, filter):
        return tf.keras.Sequential([tf.keras.layers.Conv2D(filter, 1, (1, 1), 'valid', use_bias=False),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(max_value=6)])

    def call(self, x):
        # Stage1
        x = reduce(lambda x, n: self.features.layers[n](x), list(range(0,2)), x)
        enc2x = x
        # Stage2
        x = reduce(lambda x, n: self.features.layers[n](x), list(range(2,4)), x)
        enc4x = x
        # Stage3
        x = reduce(lambda x, n: self.features.layers[n](x), list(range(4,7)), x)
        enc8x = x
        # Stage4
        x = reduce(lambda x, n: self.features.layers[n](x), list(range(7,14)), x)
        enc16x = x
        # Stage5
        x = reduce(lambda x, n: self.features.layers[n](x), list(range(14,19)), x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]


class MobileNetV2Dec(tf.keras.models.Model):
    def __init__(self):
        super(MobileNetV2Dec, self).__init__()

        expansion = 6
        filter = 32
        first_filter = 320
        interverted_residual_setting = [
            # t, c, n, s
            [1        , 16, 1, 2],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 1],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]

        # building deepest layer
        self.features = tf.keras.Sequential([self.conv_1x1_bn(first_filter)])

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting[::-1]:
            for i in range(n):
                if i == 0:
                    self.features.add(InvertedResidual(filter, c, s, expansion=t))
                else:
                    self.features.add(InvertedResidual(filter, c, 1, expansion=t))
                filter = c

        # building shallow layer
        self.features.add(self.conv_bn(3, 2))

    def conv_bn(self, filter, stride):
        return tf.keras.Sequential([tf.keras.layers.Conv2DTranspose(filter, 3, (stride, stride), 'same', use_bias=False),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(max_value=6)])

    def conv_1x1_bn(self, filter):
        return tf.keras.Sequential([tf.keras.layers.Conv2D(filter, 1, (1, 1), 'valid', use_bias=False),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(max_value=6)])

    def genLayers(self):
        # Stage1
        layer1 = tf.keras.Sequential(self.features.layers[0:8])
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(0,8)), x)

        # Stage2
        layer2 = tf.keras.Sequential(self.features.layers[8:15])
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(8,15)), x)

        # Stage3
        layer3 = tf.keras.Sequential(self.features.layers[15:17])
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(15,17)), x)

        # Stage4
        layer4 = tf.keras.Sequential(self.features.layers[17:18])
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(17,18)), x)

        # Stage5
        layer5 = tf.keras.Sequential(self.features.layers[18:19])
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(18,19)), x)

        return layer1, layer2, layer3, layer4, layer5


if(__name__=="__main__"):

    # model = MobileNetV2()
    # input = tf.zeros([32, 512, 512, 3], dtype=np.float32)
    # x = model(input)

    # layer1, layer2, layer3, layer4, layer5 = model.genLayers()
    # x = layer1(input)
    # x = layer2(x)
    # x = layer3(x)
    # x = layer4(x)
    # x = layer5(x)

    # print(x.shape)

    model = MobileNetV2Dec()
    layer1, layer2, layer3, layer4, layer5 = model.genLayers()

    input = tf.zeros([32, 16, 16, 1280], dtype=np.float32)
    x = layer1(input)
    x = layer2(x)
    x = layer3(x)
    x = layer4(x)
    x = layer5(x)

    print(x.shape)
