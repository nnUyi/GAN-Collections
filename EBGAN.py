import tensorflow as tf
import numpy as np
import time
import os
from ops import *
from glob import glob

class EBGAN:
    model_name = 'EBGAN'
    
    def __init__(self, input_height=64, input_width=64, input_channels=1, output_height=64, output_width=64, gf_dim=64, df_dim=64, batchsize=64, z_dim = 100, is_crop=False, learning_rate=0.001 , beta1=0.5, input_fname_pattern = '*.jpg', is_grayscale=False, dataset_name = 'celebA', checkpoint_dir = './checkpoint', sample_dir = 'sample', epoch = 30, sess=None):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.input_fname_pattern = input_fname_pattern      
        self.is_grayscale = is_grayscale
        self.is_crop = is_crop
        
        self.output_height = output_height
        self.output_width = output_width
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.batchsize = batchsize
        self.z_dim = z_dim
        self.beta1 = beta1
        self.learning_rate = learning_rate
        
        self.margin = 5
        self.PT_weight = 0.1
        
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.sess = sess

    def encoder(self, input_data_x, is_training=True, reuse=False):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()
            # discriminator, cnn structure
            # shape is the size of the filter
            # hidden layer_1
            shape1 = [5, 5, self.input_channels, self.df_dim]
            shape2 = [5, 5, self.df_dim, self.df_dim*2]
            shape3 = [5, 5, self.df_dim*2, self.df_dim*4]
            shape4 = [5, 5, self.df_dim*4, self.df_dim*8]
            
            # hidden layer_2                        
            h_conv1 = tf.nn.relu(conv2d(input_data_x, shape1, scope_name='d_conv1'))
            print("h_conv2_1:", h_conv1)
            
            # hidden layer_2            
            h_conv2 = tf.nn.relu(batch_norm(conv2d(h_conv1, shape2, scope_name='d_conv2'), is_training=is_training, name='d_bn_conv2'))
            print("h_conv2_2:", h_conv2)
            
            # hidden layer_3
            h_conv3 = tf.nn.relu(batch_norm(conv2d(h_conv2, shape3, scope_name='d_conv3'), is_training=is_training, name='d_bn_conv3'))
            print("h_conv2_3:", h_conv3)
            
	    # hidden layer_4
            h_conv4 = tf.nn.relu(batch_norm(conv2d(h_conv3, shape4, scope_name='d_conv4'), is_training=is_training, name='d_bn_conv4'))
            print("h_conv2_4:", h_conv4)
            
            shape_h_conv4 = h_conv4.get_shape()
            h_conv4_flat = tf.reshape(h_conv4, [self.batchsize, -1])
            h_fc1 = tf.nn.relu(batch_norm(linear(h_conv4_flat, 100, scope_name='d_fc1'), is_training=is_training, name='d_bn_fc1'))
            
            return h_fc1
    
    def decoder(self, noise_z, is_training=True, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            # auto-encoder structure
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            fc1_bn = tf.nn.relu(batch_norm(linear(noise_z, s_h8*s_w8*self.gf_dim*8, scope_name='d_dc_fc1'), is_training=is_training, name='d_dc_fc1_bn'))
            
            fc2_deconv = tf.reshape(fc1_bn, [-1, s_h8, s_w8, self.gf_dim*8])
            print("deconv2d_1:", fc2_deconv)
            
            # deconv layer_2
            filter_shape2 = [5, 5, self.gf_dim*4, self.gf_dim*8]
            output_shape2 = [self.batchsize, s_h4, s_w4, self.gf_dim*4]
            h_deconv2 = tf.nn.relu(batch_norm(deconv2d(fc2_deconv, filter_shape2, output_shape2, scope_name='d_deconv2'), is_training=is_training, name='d_bn_deconv2'))
            print("deconv2d_2:",h_deconv2)
            
	    # deconv layer_3
            filter_shape3 = [5,5,self.gf_dim*2, self.gf_dim*4]
            output_shape3 = [self.batchsize, s_h2, s_w2, self.gf_dim*2]
            h_deconv3 = tf.nn.relu(batch_norm(deconv2d(h_deconv2, filter_shape3, output_shape3, scope_name='d_deconv3'), is_training=is_training, name='d_bn_deconv3'))
            print("deconv2d_3:", h_deconv3)
	
	    # deconv layer_4
            filter_shape4 = [5,5, self.input_channels, self.gf_dim*2]
            output_shape4 = [self.batchsize, s_h, s_w, self.input_channels]
            h_deconv4 = tf.nn.tanh(deconv2d(h_deconv3, filter_shape4, output_shape4, scope_name='d_deconv4'))
            print("deconv2d_4:", h_deconv4)            
            
            return h_deconv4

    def discriminator(self, input_data_x, is_training=True, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            hidden_code = self.encoder(input_data_x, is_training, reuse)
            reconstr = self.decoder(hidden_code, is_training, reuse)    
            reconstr_error = tf.sqrt(tf.reduce_sum(tf.square(input_data_x-reconstr)))/self.batchsize
        
            return reconstr, reconstr_error, hidden_code

    def pull_away(self, hidden_code):
        norm = tf.sqrt(tf.reduce_sum(tf.square(hidden_code), 1, keep_dims=True))
        normalized_hidden = hidden_code / norm

        similarity = tf.matmul(normalized_hidden, normalized_hidden, transpose_b=True)
        batchsize = tf.cast(self.batchsize, tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batchsize) / (batchsize * (batchsize - 1))
        return pt_loss
    
    def generator(self, noise_z, is_training=True, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            # auto-encoder structure
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            fc1_bn = tf.nn.relu(batch_norm(linear(noise_z, s_h8*s_w8*self.gf_dim*8, scope_name='g_fc1'), is_training=is_training, name='g_fc1_bn'))
            
            fc2_deconv = tf.reshape(fc1_bn, [-1, s_h8, s_w8, self.gf_dim*8])
            print("deconv2d_1:", fc2_deconv)
            
            # deconv layer_2
            filter_shape2 = [5, 5, self.gf_dim*4, self.gf_dim*8]
            output_shape2 = [self.batchsize, s_h4, s_w4, self.gf_dim*4]
            h_deconv2 = tf.nn.relu(batch_norm(deconv2d(fc2_deconv, filter_shape2, output_shape2, scope_name='g_deconv2'), is_training=is_training, name='g_bn_deconv2'))
            print("deconv2d_2:",h_deconv2)
            
            # deconv layer_3
            filter_shape3 = [5,5,self.gf_dim*2, self.gf_dim*4]
            output_shape3 = [self.batchsize, s_h2, s_w2, self.gf_dim*2]
            h_deconv3 = tf.nn.relu(batch_norm(deconv2d(h_deconv2, filter_shape3, output_shape3, scope_name='g_deconv3'), is_training=is_training, name='g_bn_deconv3'))
            print("deconv2d_3:", h_deconv3)
	
	    # deconv layer_4
            filter_shape4 = [5,5, self.input_channels, self.gf_dim*2]
            output_shape4 = [self.batchsize, s_h, s_w, self.input_channels]
            h_deconv4 = tf.nn.tanh(deconv2d(h_deconv3, filter_shape4, output_shape4, scope_name='g_deconv4'))
            print("deconv2d_4:", h_deconv4)            
            
            return h_deconv4
            
    def build_model(self):
        img_dims = [self.input_height, self.input_width, self.input_channels]
        
        self.input_data = tf.placeholder(tf.float32, [self.batchsize] + img_dims, name='real_data')
        self.z = tf.placeholder(tf.float32, [self.batchsize, self.z_dim], name='z')

        real_reconstr, real_reconstr_error, real_hidden_code = self.discriminator(self.input_data, is_training=True, reuse=False)
        
        sample = self.generator(self.z, is_training=True, reuse=False)
        fake_reconstr, fake_reconstr_error, fake_hidden_code = self.discriminator(sample, is_training=True, reuse=True)
        
        PT = self.pull_away(fake_hidden_code)
        
        self.d_loss = real_reconstr_error + tf.maximum(self.margin - fake_reconstr_error, 0)
        self.g_loss = fake_reconstr_error + self.PT_weight*PT        
        
        t_vars = tf.trainable_variables()
        
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        self.sample_images = self.generator(self.z, is_training=False, reuse=True)
        
        #self.d_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        #self.g_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        #self.encoder_optimization = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.loss)
        self.d_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        # saver for saving model
        self.saver = tf.train.Saver()

    def train(self):        
        try:
            tf.global_variables_initializer().run()
        except AttributeError:
            tf.initialize_all_variables().run()
        # sample real_images and noise_z for testing
        sample_data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
        print(len(sample_data))
        sample_files = sample_data[0:self.batchsize]
        sample_batch_x = [get_image(sample_file,is_grayscale=self.is_grayscale) for sample_file in sample_files]
        
        if (self.is_grayscale):
            sample_batch_x = np.array(sample_batch_x).astype(np.float32)[:, :, :, None]
        else:
            sample_batch_x = np.array(sample_batch_x).astype(np.float32)

        sample_z = np.random.normal(0,1, [self.batchsize, self.z_dim]).astype(np.float32)
        sample_batch_x = 2*((sample_batch_x/255.) - 0.5)

        counter_bool, counter = self.load(self.checkpoint_dir)
        if counter_bool:
            counter = counter + 1
            print("[***]load model successfully")
        else:
            counter = 1
            print("[***]fail to load model")
        start_time = time.time()
        for index in range(self.epoch):
            # code just for images datasets
            data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            batch_idxs = int(len(data)/self.batchsize)
            
            for idx in range(batch_idxs):
                batch_files = data[idx*self.batchsize:(idx+1)*self.batchsize]
                # load data from datasets
                batch = [get_image(batch_file, is_grayscale=self.is_grayscale) for batch_file in batch_files]
                
                if (self.is_grayscale):
                    batch_x = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_x = np.array(batch).astype(np.float32)
                # normalization
                batch_x = 2*((batch_x/255.)-0.5)
                batch_z = np.random.normal(0,1, [self.batchsize, self.z_dim]).astype(np.float32)
                
                # update discriminator
                _, d_loss = self.sess.run([self.d_optimization, self.d_loss], feed_dict={self.input_data:batch_x, self.z:batch_z})
                # update generator
                _, g_loss = self.sess.run([self.g_optimization, self.g_loss], feed_dict={self.z:batch_z})


                iteration_time = time.time()
                total_time = (iteration_time - start_time)
                print("epoch[%d]:[%d/%d]: " %(index, idx, batch_idxs), "total_time:{:.4f}".format(total_time), "d_loss:{:.4f},g_loss:{:.4f}".format(d_loss, g_loss))

                counter = counter + 1
                if np.mod(idx, 100) == 0:
                    iteration_time = time.time()
                    total_time = (iteration_time - start_time)
                    # sample images and save them
                    samples = self.sess.run(self.sample_images, feed_dict={self.z:sample_z})
                    #print(samples)
                    save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, index, idx))
                    # calc loss
                    sample_d_loss, sample_g_loss = self.sess.run([self.d_loss, self.g_loss], feed_dict={self.input_data:sample_batch_x, self.z:sample_z})
                                                               
                    print("epoch[%d]:[%d/%d]: " %(index, idx, batch_idxs), "total_time:{:.4f}".format(total_time), "d_loss:{:.4f},g_loss:{:.4f}".format(sample_d_loss, sample_g_loss))

                # save model
                if np.mod(counter, 500) == 0:
                    self.save_model(self.checkpoint_dir, counter)
    
    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batchsize, self.z_dim)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
