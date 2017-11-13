import tensorflow as tf
import os
from WGAN_GP import *

flags = tf.app.flags
flags.DEFINE_bool("is_training",True, "training flags")
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists('./sample'):
        os.mkdir('./sample')
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

if __name__=='__main__':
    check_dir()
    with tf.Session() as sess:
        GAN = WGAN_GP(input_height=64, input_width=64, input_channels=3, output_height=64, output_width=64, gf_dim=64, input_fname_pattern = '*.jpg', is_grayscale=False, sess = sess)
        GAN.build_model()
        if FLAGS.is_training:
            GAN.train()
