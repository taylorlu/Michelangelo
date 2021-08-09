import tensorflow as tf
import numpy as np
import os, sys, cv2
from tqdm import trange
from generator.generator import DataGenerator
from michelangelo.network import UNetModel
from michelangelo.scheduling import piecewise_linear_schedule

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if(__name__=='__main__'):
    
    model = UNetModel()

    train_dataset = DataGenerator()
    train_dataset.prepare_dataset()

    losses = []
    t = trange(model.step, 100000, leave=True)
    for _ in t:
        t.set_description(f'step {model.step}')
        image, fg, bg, alpha, image_name = train_dataset.next_batch()
        # learning_rate = piecewise_linear_schedule(model.step, config_dict['learning_rate_schedule'])
        # model.set_constants(learning_rate=learning_rate)
        model._compile()

        output = model.train_step(inputs=image,
                                  fg=fg,
                                  bg=bg,
                                  alpha=alpha,
                                  image_name=image_name)

        # losses.append(float(output['loss']))
