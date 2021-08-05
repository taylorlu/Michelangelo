import tensorflow as tf
import numpy as np
from mobilenetv2 import MobileNetV2, MobileNetV2Dec


class BasicBlock(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, stride=1):
        super(BasicBlock, self).__init__()

        if(stride>1):
            self.conv1 = tf.keras.layers.Conv2DTranspose(filterIn // 2, 4, (2, 2), 'same', use_bias=False)
        else:
            self.conv1 = tf.keras.layers.Conv2D(filterIn // 2, 3, (1, 1), 'same', use_bias=False)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leakyRelu = tf.keras.layers.LeakyReLU(0.2)
        self.conv2 = tf.keras.layers.Conv2D(filterOut, 3, (1, 1), 'same', use_bias=False)

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRelu(out)
        out = self.conv2(out)
        return out


class SEBlock(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.Sequential([tf.keras.layers.Dense(int(filterIn // reduction), use_bias=False),
                                       tf.keras.layers.ReLU(),
                                       tf.keras.layers.Dense(filterOut, use_bias=False, activation='sigmoid')])

    def call(self, x):
        w = self.pool(x)
        w = self.fc(w)
        w = tf.expand_dims(tf.expand_dims(w, axis=1), axis=1)
        return x * w


class ASPP(tf.keras.models.Model):
    def __init__(self, filter):
        super(ASPP, self).__init__()
        mid_filter = 128
        dilations = [(1, 1), (2, 2), (4, 4), (8, 8)]

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.relu = tf.keras.layers.ReLU()
        self.aspp1 = tf.keras.layers.Conv2D(mid_filter, 1, (1, 1), 'same', dilation_rate=dilations[0], use_bias=False)
        self.aspp2 = tf.keras.layers.Conv2D(mid_filter, 3, (1, 1), 'same', dilation_rate=dilations[1], use_bias=False)
        self.aspp3 = tf.keras.layers.Conv2D(mid_filter, 3, (1, 1), 'same', dilation_rate=dilations[2], use_bias=False)
        self.aspp4 = tf.keras.layers.Conv2D(mid_filter, 3, (1, 1), 'same', dilation_rate=dilations[3], use_bias=False)
        self.aspp5 = tf.keras.layers.Conv2D(mid_filter, 1, (1, 1), 'same', use_bias=False)

        self.aspp1_bn = tf.keras.layers.BatchNormalization()
        self.aspp2_bn = tf.keras.layers.BatchNormalization()
        self.aspp3_bn = tf.keras.layers.BatchNormalization()
        self.aspp4_bn = tf.keras.layers.BatchNormalization()
        self.aspp5_bn = tf.keras.layers.BatchNormalization()

        self.conv = tf.keras.layers.Conv2D(filter, 1, (1, 1), 'same', use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.pool(x)
        x5 = tf.expand_dims(tf.expand_dims(x5, 1), 1)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = tf.keras.layers.UpSampling2D(size=(tf.shape(x)[1], tf.shape(x)[2]))(x5)
        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MainBranch(tf.keras.models.Model):
    def __init__(self):
        super(MainBranch, self).__init__()

        self.encoder = MobileNetV2()
        enc_filters = self.encoder.enc_filters
        print(enc_filters)

        self.aspp = ASPP(enc_filters[4])
        self.se_block = SEBlock(enc_filters[4], enc_filters[4], reduction=4)

        self.decoder = MobileNetV2Dec()
        layers = self.decoder.genLayers()
        self.layer1 = layers[0]
        self.layer2 = layers[1]
        self.layer3 = layers[2]
        self.layer4 = layers[3]
        self.layer5 = layers[4]

        self.refine_OS1 = BasicBlock(enc_filters[0], 1)
        self.refine_OS4 = BasicBlock(enc_filters[1], 1)
        self.refine_OS8 = BasicBlock(enc_filters[2], 1)
        self.up8x8 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')
        self.up4x4 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')

    def call(self, img):
        enc_features = self.encoder(img)
        fea1, fea2, fea3, fea4, fea5 = enc_features

        x = self.aspp(fea5)
        x = self.se_block(x)

        x = self.layer1(x) + fea4
        x = self.layer2(x) + fea3   #(None, 64, 64, 32)
        x_os8 = self.refine_OS8(x)  #(None, 64, 64, 1)
        x_os8 = self.up8x8(x_os8)   #(None, 512, 512, 1)

        x = self.layer3(x) + fea2   #(None, 128, 128, 24)
        x_os4 = self.refine_OS4(x)  #(None, 128, 128, 1)
        x_os4 = self.up4x4(x_os4)   #(None, 512, 512, 1)

        x = self.layer4(x) + fea1   #(None, 256, 256, 16)
        x = self.layer5(x)          #(None, 512, 512, 3)
        x_os1 = self.refine_OS1(x)  #(None, 512, 512, 1)

        x_os1 = (tf.nn.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (tf.nn.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (tf.nn.tanh(x_os8) + 1.0) / 2.0

        return x_os1, x_os4, x_os8


if(__name__=='__main__'):

    x = tf.zeros([16, 512, 512, 3],  dtype=tf.float32)

    branch = MainBranch()
    x = branch(x)
    # print(x.shape)
