import tensorflow as tf
import numpy as np
import os, sys, cv2
from tqdm import trange
from generator.generator import DataGenerator
from michelangelo.network import UNetModel
from michelangelo.scheduling import piecewise_linear_schedule
import traceback


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

def dynamic_memory_allocation2():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

dynamic_memory_allocation()


if(__name__=='__main__'):
    
    model = UNetModel()
    model._compile()

    train_dataset = DataGenerator()
    train_dataset.prepare_dataset()

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                    optimizer=model.optimizer,
                                    net=model)
    manager = tf.train.CheckpointManager(checkpoint, 'latest',
                                        max_to_keep=5,
                                        keep_checkpoint_every_n_hours=5)
    checkpoint.restore(manager.latest_checkpoint)

    image = cv2.imread('test.jpg')/255.0
    image = cv2.resize(image, (512, 512))[np.newaxis, ...]

    output = model.test_step(inputs=image)

    cv2.imwrite(f'output/eval_os1.jpg', (output['alpha_pred_os1'].numpy()[0, ...]*255).astype(np.uint8))
    cv2.imwrite(f'output/eval_os4.jpg', (output['alpha_pred_os4'].numpy()[0, ...]*255).astype(np.uint8))
    cv2.imwrite(f'output/eval_os8.jpg', (output['alpha_pred_os8'].numpy()[0, ...]*255).astype(np.uint8))
