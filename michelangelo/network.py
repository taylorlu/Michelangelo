import tensorflow as tf
import numpy as np
from michelangelo.mobilenetv2 import MobileNetV2, MobileNetV2Dec, while_loop
from michelangelo.losses import regression_loss, composition_loss, lap_loss, get_unknown_tensor_from_pred
# from mobilenetv2 import MobileNetV2, MobileNetV2Dec
# from losses import regression_loss, composition_loss, lap_loss, get_unknown_tensor_from_pred


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

    def call(self, x, training=True):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.leakyRelu(out)
        out = self.conv2(out)
        return out


class SEBlock(tf.keras.models.Model):
    def __init__(self, filterIn, filterOut, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = [tf.keras.layers.Dense(int(filterIn // reduction), use_bias=False),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(filterOut, use_bias=False, activation='sigmoid')]

    def call(self, x, training=True):
        w = self.pool(x)
        w = while_loop(self.fc, w, training)
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

    def call(self, x, training=True):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1, training=training)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2, training=training)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3, training=training)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4, training=training)
        x4 = self.relu(x4)
        x5 = self.pool(x)
        x5 = tf.expand_dims(tf.expand_dims(x5, 1), 1)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5, training=training)
        x5 = self.relu(x5)
        x5 = tf.keras.layers.UpSampling2D(size=(16, 16))(x5)
        x = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class MainBranch(tf.keras.models.Model):
    def __init__(self):
        super(MainBranch, self).__init__()

        self.encoder = MobileNetV2()
        enc_filters = self.encoder.enc_filters

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

    def call(self, img, training=True):
        enc_features = self.encoder(img, training)
        fea1, fea2, fea3, fea4, fea5 = enc_features

        x = self.aspp(fea5, training)
        x = self.se_block(x, training)

        x = while_loop(self.layer1, x, training) + fea4
        x = while_loop(self.layer2, x, training) + fea3   #(None, 64, 64, 32)
        x_os8 = self.refine_OS8(x, training)  #(None, 64, 64, 1)
        x_os8 = self.up8x8(x_os8)   #(None, 512, 512, 1)

        x = while_loop(self.layer3, x, training) + fea2   #(None, 128, 128, 24)
        x_os4 = self.refine_OS4(x, training)  #(None, 128, 128, 1)
        x_os4 = self.up4x4(x_os4)   #(None, 512, 512, 1)

        x = while_loop(self.layer4, x, training) + fea1   #(None, 256, 256, 16)
        x = while_loop(self.layer5, x, training)          #(None, 512, 512, 3)
        x_os1 = self.refine_OS1(x, training)  #(None, 512, 512, 1)

        x_os1 = (tf.nn.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (tf.nn.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (tf.nn.tanh(x_os8) + 1.0) / 2.0

        return x_os1, x_os4, x_os8


class UNetModel(tf.keras.models.Model):

    def __init__(self, debug=False, **kwargs):
        super(UNetModel, self).__init__(**kwargs)
        self.mainBranch = MainBranch()

        self.training_input_signature = [
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 512, 512, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.string)
        ]
        self.test_input_signature = [
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32)
        ]
        self.debug = debug
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate,
                                                beta_1=0.9,
                                                beta_2=0.98,
                                                epsilon=1e-9)
        self._apply_all_signatures()

    @property
    def step(self):
        return int(self.optimizer.iterations)

    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)

    def _apply_all_signatures(self):
        self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
        self.test_step = self._apply_signature(self._test_step, self.test_input_signature)

    def _train_step(self, inputs, fg, bg, alpha, image_name):
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs, training=True)

            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = model_out['x_os1'], model_out['x_os4'], model_out['x_os8']

            weight_os8 = tf.ones_like(alpha_pred_os8, dtype=np.float32)
            weight_os4 = get_unknown_tensor_from_pred(alpha_pred_os8)
            weight_os1 = get_unknown_tensor_from_pred(alpha_pred_os4)

            self.loss_dict = {}
            self.loss_dict['rec'] = (self.loss[0](alpha_pred_os8, alpha, weight=weight_os8) *1) +\
                                    (self.loss[0](alpha_pred_os4, alpha, weight=weight_os4) *1) +\
                                    (self.loss[0](alpha_pred_os1, alpha, weight=weight_os1) *2)

            self.loss_dict['comp'] = (self.loss[1](alpha_pred_os8, fg, bg, inputs, weight=weight_os8) *1) +\
                                     (self.loss[1](alpha_pred_os4, fg, bg, inputs, weight=weight_os4) *1) +\
                                     (self.loss[1](alpha_pred_os1, fg, bg, inputs, weight=weight_os1) *2)

            self.loss_dict['lap'] = (self.loss[2](alpha_pred_os8, alpha, weight=weight_os8) *1) +\
                                    (self.loss[2](alpha_pred_os4, alpha, weight=weight_os4) *1) +\
                                    (self.loss[2](alpha_pred_os1, alpha, weight=weight_os1) *2)

            loss = 0
            for loss_key in self.loss_dict.keys():
                if self.loss_dict[loss_key] is not None and loss_key in ['rec', 'comp', 'lap']:
                    loss += self.loss_dict[loss_key]

            model_out.update({'loss': loss})
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return model_out

    def _test_step(self, inputs):
        with tf.GradientTape() as tape:
            model_out = self.__call__(inputs, training=False)

            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = model_out['x_os1'], model_out['x_os4'], model_out['x_os8']

            model_out.update({'alpha_pred_os1': alpha_pred_os1})
            model_out.update({'alpha_pred_os4': alpha_pred_os4})
            model_out.update({'alpha_pred_os8': alpha_pred_os8})

        return model_out

    def call(self, inputs, training=True):
        x_os1, x_os4, x_os8 = self.mainBranch(inputs, training)
        model_out = {}
        model_out.update({'x_os1': x_os1, 'x_os4': x_os4, 'x_os8': x_os8})
        return model_out

    def _compile(self):
        self.loss_weights = [1., 1., 1.]
        self.compile(loss=[regression_loss, composition_loss, lap_loss],
                     loss_weights=self.loss_weights,
                     optimizer=self.optimizer)

    def set_constants(self, learning_rate: float = None):
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)


if(__name__=='__main__'):

    x = tf.zeros([16, 512, 512, 3],  dtype=tf.float32)

    branch = MainBranch()
    x = branch(x)
    # print(x.shape)
