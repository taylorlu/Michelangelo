import tensorflow as tf
import numpy as np
import os, sys, cv2
from tqdm import trange
from generator.generator import DataGenerator
from network import UNetModel
from scheduling import piecewise_linear_schedule

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if(__name__=='__main__'):
    
    model = UNetModel()

    train_dataset = DataGenerator()
    train_dataset.prepare_dataset()

    losses = []
    t = trange(model.step, 100000, leave=True)
    for _ in t:
        t.set_description(f'step {model.step}')
        spk, mel, phonemes, mel_len, phon_len, fname = train_dataset.next_batch()
        learning_rate = piecewise_linear_schedule(model.step, config_dict['learning_rate_schedule'])
        model.set_constants(learning_rate=learning_rate)

        output = model.train_step(spk=spk,
                                mel_inp=mel,
                                phon_tar=phonemes,
                                mel_inp_len=mel_len,
                                phon_tar_len=phon_len)
        losses.append(float(output['loss']))
