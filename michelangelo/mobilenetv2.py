import tensorflow as tf
import numpy as np
from functools import reduce


def while_loop(layer, x, training=True):
    if(isinstance(layer, list)):
        for l in layer:
            x = while_loop(l, x)
    else:
        x = layer(x, training=training)
    return x


class InvertedResidual(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, stride, expansion):
        super(InvertedResidual, self).__init__()
        hidden_dim = filterIn * expansion
        self.convlist = []
        self.use_res_connect = stride == 1 and filterIn == filterOut

        if(expansion==1):
            self.convlist.append(tf.keras.layers.UpSampling2D(size=(stride, stride), interpolation='bilinear'))
            self.convlist.append(tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False))
            self.convlist.append(tf.keras.layers.Conv2D(hidden_dim, 1, (1, 1), 'same', groups=hidden_dim, use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
            self.convlist.append(tf.keras.layers.ReLU(max_value=6))
            self.convlist.append(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
        else:
            self.convlist.append(tf.keras.layers.Conv2D(hidden_dim, 1, (1, 1), 'valid', use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
            self.convlist.append(tf.keras.layers.ReLU(max_value=6))
            self.convlist.append(tf.keras.layers.UpSampling2D(size=(stride, stride), interpolation='bilinear'))
            self.convlist.append(tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False))
            self.convlist.append(tf.keras.layers.Conv2D(hidden_dim, 1, (1, 1), 'same', groups=hidden_dim, use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
            self.convlist.append(tf.keras.layers.ReLU(max_value=6))
            self.convlist.append(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())

    def call(self, x, training=True):
        if(self.use_res_connect):
            current = x
            for conv in self.convlist:
                x = conv(x, training=training)
            x += current
            return x
        else:
            for conv in self.convlist:
                x = conv(x, training=training)
            return x


class Residual(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, stride, expansion):
        super(Residual, self).__init__()
        hidden_dim = filterIn * expansion
        self.convlist = []
        self.use_res_connect = stride == 1 and filterIn == filterOut

        if(expansion==1):
            self.convlist.append(tf.keras.layers.Conv2D(hidden_dim, 3, (stride, stride), 'same', groups=hidden_dim, use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
            self.convlist.append(tf.keras.layers.ReLU(max_value=6))
            self.convlist.append(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
        else:
            self.convlist.append(tf.keras.layers.Conv2D(hidden_dim, 1, (1, 1), 'valid', use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
            self.convlist.append(tf.keras.layers.ReLU(max_value=6))
            self.convlist.append(tf.keras.layers.Conv2D(hidden_dim, 3, (stride, stride), 'same', groups=hidden_dim, use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())
            self.convlist.append(tf.keras.layers.ReLU(max_value=6))
            self.convlist.append(tf.keras.layers.Conv2D(filterOut, 1, (1, 1), 'valid', use_bias=False))
            self.convlist.append(tf.keras.layers.BatchNormalization())

    def call(self, x, training=True):
        if(self.use_res_connect):
            current = x
            for conv in self.convlist:
                x = conv(x, training=training)
            x += current
            return x
        else:
            for conv in self.convlist:
                x = conv(x, training=training)
            return x


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
        self.features = [self.conv_bn(filter, 2)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            for i in range(n):
                if i == 0:
                    self.features.append(Residual(filter, c, s, expansion=t))
                else:
                    self.features.append(Residual(filter, c, 1, expansion=t))
                filter = c

        # building last several layers
        self.features.append(self.conv_1x1_bn(last_filter))

    def conv_bn(self, filter, stride):
        return [tf.keras.layers.Conv2D(filter, 3, (stride, stride), 'same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6)]

    def conv_1x1_bn(self, filter):
        return [tf.keras.layers.Conv2D(filter, 1, (1, 1), 'valid', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6)]

    def call(self, x, training=True):

        # Stage1
        x = reduce(lambda x, n: while_loop(self.features[n], x, training), list(range(0,2)), x)
        enc2x = x
        # Stage2
        x = reduce(lambda x, n: while_loop(self.features[n], x, training), list(range(2,4)), x)
        enc4x = x
        # Stage3
        x = reduce(lambda x, n: while_loop(self.features[n], x, training), list(range(4,7)), x)
        enc8x = x
        # Stage4
        x = reduce(lambda x, n: while_loop(self.features[n], x, training), list(range(7,14)), x)
        enc16x = x
        # Stage5
        x = reduce(lambda x, n: while_loop(self.features[n], x, training), list(range(14,19)), x)
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
        self.features = [self.conv_1x1_bn(first_filter)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting[::-1]:
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(filter, c, s, expansion=t))
                else:
                    self.features.append(InvertedResidual(filter, c, 1, expansion=t))
                filter = c

        # building shallow layer
        self.features.append(self.conv_bn(3, 2))

    def conv_bn(self, filter, stride):
        return [tf.keras.layers.Conv2DTranspose(filter, 3, (stride, stride), 'same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6)]

    def conv_1x1_bn(self, filter):
        return [tf.keras.layers.Conv2D(filter, 1, (1, 1), 'valid', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(max_value=6)]

    def genLayers(self):
        # Stage1
        layer1 = self.features[0:8]
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(0,8)), x)

        # Stage2
        layer2 = self.features[8:15]
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(8,15)), x)

        # Stage3
        layer3 = self.features[15:17]
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(15,17)), x)

        # Stage4
        layer4 = self.features[17:18]
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(17,18)), x)

        # Stage5
        layer5 = self.features[18:19]
        # x = reduce(lambda x, n: self.features.layers[n](x), list(range(18,19)), x)

        return layer1, layer2, layer3, layer4, layer5


if(__name__=="__main__"):

    model = MobileNetV2()
    input = tf.zeros([32, 512, 512, 3], dtype=np.float32)

    x1, x2, x3, x4, x5 = model(input)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)

    model = MobileNetV2Dec()
    layer1, layer2, layer3, layer4, layer5 = model.genLayers()

    input = tf.zeros([32, 16, 16, 1280], dtype=np.float32)
    x = while_loop(layer1, input)
    x = while_loop(layer2, x)
    x = while_loop(layer3, x)
    x = while_loop(layer4, x)
    x = while_loop(layer5, x)

    print(x.shape)
