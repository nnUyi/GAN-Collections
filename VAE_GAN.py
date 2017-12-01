import tensorflow as tf
import numpy as np
import time
import os
from ops import *
from glob import glob

class VAE_GAN:
    model_name = 'VAE_GAN'
    
    def __init__(self, input_height=64, input_width=64, input_channels=1, output_height=64, output_width=64, gf_dim=64, df_dim=64, batchsize=64, z_dim = 100, is_crop=False, learning_rate=3e-4 , beta1=0.5, input_fname_pattern = '*.jpg', is_grayscale=False, dataset_name = 'celebA', checkpoint_dir = './checkpoint', sample_dir = 'sample', epoch = 30, sess=None):
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
        
        self.epsilon = 1e-8
        self.gamma = 1e-6
        
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
            
            # hidden layer_2                        
            h_conv1 = tf.nn.relu(conv2d(input_data_x, shape1, scope_name='e_conv1'))
            print("h_conv2_1:", h_conv1)
            
            # hidden layer_2            
            h_conv2 = tf.nn.relu(batch_norm(conv2d(h_conv1, shape2, scope_name='e_conv2'), is_training=is_training, name='e_bn_conv2'))
            print("h_conv2_2:", h_conv2)
            
            # hidden layer_3
            h_conv3 = tf.nn.relu(batch_norm(conv2d(h_conv2, shape3, scope_name='e_conv3'), is_training=is_training, name='e_bn_conv3'))
            print("h_conv2_3:", h_conv3)
            
            shape_h_conv3 = h_conv3.get_shape()
            h_conv3_flat = tf.reshape(h_conv3, [self.batchsize, -1])            
            h_fc1 = tf.nn.relu(batch_norm(linear(h_conv3_flat, 2048, scope_name='e_fc1'), is_training=is_training, name='e_bn_fc1'))
            
            # hidden layer_4 fully connected
            mu_sigma = linear(h_fc1, 2*self.z_dim, scope_name='e_fc2')
            
            mean = mu_sigma[:,0:self.z_dim]
            sigma = mu_sigma[:, self.z_dim:2*self.z_dim]
            sigma = tf.nn.softplus(sigma)
            
            return mean, sigma
    
    def decoder(self, noise_z, is_training=True, reuse=False):
        with tf.variable_scope('decoder') as scope:
            if reuse:
                scope.reuse_variables()
            # auto-encoder structure
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            fc1_bn = tf.nn.relu(batch_norm(linear(noise_z, s_h16*s_w16*self.gf_dim*4, scope_name='g_fc1'), is_training=is_training, name='g_fc1_bn'))
            
            fc2_deconv = tf.reshape(fc1_bn, [-1, s_h16, s_w16, self.gf_dim*4])
            print("deconv2d_1:", fc2_deconv)
            
	    # deconv layer_2
            filter_shape2 = [5, 5, self.gf_dim*4, self.gf_dim*4]
            output_shape2 = [self.batchsize, s_h8, s_w8, self.gf_dim*4]
            h_deconv2 = tf.nn.relu(batch_norm(deconv2d(fc2_deconv, filter_shape2, output_shape2, scope_name='g_deconv2'), is_training=is_training, name='g_bn_deconv2'))
            print("deconv2d_2:",h_deconv2)
            
	    # deconv layer_3
            filter_shape3 = [5,5,self.gf_dim*2, self.gf_dim*4]
            output_shape3 = [self.batchsize, s_h4, s_w4, self.gf_dim*2]
            h_deconv3 = tf.nn.relu(batch_norm(deconv2d(h_deconv2, filter_shape3, output_shape3, scope_name='g_deconv3'), is_training=is_training, name='g_bn_deconv3'))
            print("deconv2d_3:", h_deconv3)
	
	    # deconv layer_4
            filter_shape4 = [5,5, 32, self.gf_dim*2]
            output_shape4 = [self.batchsize, s_h2, s_w2, 32]
            h_deconv4 = tf.nn.relu(batch_norm(deconv2d(h_deconv3, filter_shape4, output_shape4, scope_name='g_deconv4'), is_training=is_training, name='g_bn_deconv4'))
            print("deconv2d_4:", h_deconv4)
            
            # deconv layer_5
            filter_shape5 = [5,5,self.input_channels, 32]
            output_shape5 = [self.batchsize, s_h, s_w, self.input_channels]
            h_deconv5 = tf.nn.tanh(deconv2d(h_deconv4, filter_shape5, output_shape5, scope_name='g_deconv5'))
            print("deconv2d_5", h_deconv5)

            return h_deconv5

    def discriminator(self, input_data_x, is_training=True, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            
            shape1 = [5, 5, self.input_channels, self.df_dim/2]
            shape2 = [5, 5, self.df_dim/2, self.df_dim*2]
            shape3 = [5, 5, self.df_dim*2, self.df_dim*4]
            shape4 = [5, 5, self.df_dim*4, self.df_dim*4]
            
            # hidden layer_2                        
            h_conv1 = leaky_relu(conv2d(input_data_x, shape1, scope_name='d_conv1'))
            print("h_conv2_1:", h_conv1)
            
            # hidden layer_2            
            h_conv2 = leaky_relu(batch_norm(conv2d(h_conv1, shape2, scope_name='d_conv2'), is_training=is_training, name='d_bn_conv2'))
            print("h_conv2_2:", h_conv2)
            
            # hidden layer_3
            h_conv3 = leaky_relu(batch_norm(conv2d(h_conv2, shape3, scope_name='d_conv3'), is_training=is_training, name='d_bn_conv3'))
            print("h_conv2_3", h_conv3)

            # hidden layer_4
            conv4 = conv2d(h_conv3, shape4, scope_name='d_conv4')
            h_conv4 = leaky_relu(batch_norm(conv4, is_training=is_training, name='d_bn_conv4'))
            print('h_conv2_4', h_conv4)
            
            shape_h_conv4 = h_conv4.get_shape()
            h_conv4_flat = tf.reshape(h_conv4, [self.batchsize, -1])
            h_fc1 = leaky_relu(batch_norm(linear(h_conv4_flat, 512, scope_name='d_fc1'), is_training=is_training, name='d_bn_fc1'))
            
            # hidden layer_4 fully connected
            h_fc2_sigmoid = tf.nn.sigmoid(linear(h_fc1, 1, scope_name='d_fc2'))
            
            return h_fc2_sigmoid, conv4
            
    def build_model(self):
        img_dims = [self.input_height, self.input_width, self.input_channels]
        
        self.input_data = tf.placeholder(tf.float32, [self.batchsize] + img_dims, name='real_data')
        self.z = tf.placeholder(tf.float32, [self.batchsize, self.z_dim], name='z')
        
        mean, sigma = self.encoder(self.input_data, is_training=True, reuse=False)
        z = mean + sigma*tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
        
        self.out_x_hat = self.decoder(z, is_training=True, reuse=False)
        self.out_x_fake = self.decoder(self.z, is_training=True, reuse=True)
        
        self.D_x_real_logits, self.D_x_real_l = self.discriminator(self.input_data, is_training=True, reuse=False)
        self.D_x_fake_logits, _ = self.discriminator(self.out_x_fake, is_training=True, reuse=True)
        self.D_x_hat_logits, self.D_x_hat_l = self.discriminator(self.out_x_hat, is_training=True, reuse=True)
        
        self.L_prior = tf.reduce_mean(tf.reduce_sum(tf.square(mean) + tf.square(sigma) - tf.log(tf.square(sigma) + self.epsilon) - 1, 1))
        
        self.L_llike = tf.reduce_mean(tf.square(self.D_x_real_l - self.D_x_hat_l))
        
        def sigmoid_cross_entropy_with_logits(x,y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
        
        self.D_x_real_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_x_real_logits, tf.ones_like(self.D_x_real_logits)))
        self.D_x_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_x_fake_logits, tf.zeros_like(self.D_x_fake_logits)))
        self.D_x_hat_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_x_hat_logits, tf.zeros_like(self.D_x_hat_logits)))
        
        self.G_x_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_x_fake_logits, tf.ones_like(self.D_x_fake_logits)))
        self.G_x_hat_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_x_hat_logits, tf.ones_like(self.D_x_hat_logits)))
        
        self.L_gan = self.D_x_real_loss + self.D_x_fake_loss + self.D_x_hat_loss
        
        self.encoder_loss = self.L_prior + self.L_llike
        self.decoder_loss = self.G_x_fake_loss + self.G_x_hat_loss + self.gamma*self.L_llike
        self.dis_loss = self.L_gan
        
        t_vars = tf.trainable_variables()
        
        self.encoder_vars = [var for var in t_vars if 'e_' in var.name]
        self.decoder_vars = [var for var in t_vars if 'g_' in var.name]
        self.dis_vars = [var for var in t_vars if 'd_' in var.name]
        
        self.sample_images = self.decoder(self.z, is_training=False, reuse=True)
        
        #self.d_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        #self.g_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        #self.encoder_optimization = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.loss)
        self.encoder_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.encoder_loss, var_list=self.encoder_vars)
        self.decoder_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.decoder_loss, var_list=self.decoder_vars)
        self.d_optimization = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.dis_loss, var_list=self.dis_vars)

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
                
                # update encoder
                _, encoder_loss = self.sess.run([self.encoder_optimization, self.encoder_loss], feed_dict={self.input_data:batch_x})
                # update decoder
                _, decoder_loss = self.sess.run([self.decoder_optimization, self.decoder_loss], feed_dict={self.input_data:batch_x, self.z:batch_z})
                # update discriminator
                _, dis_loss = self.sess.run([self.d_optimization, self.dis_loss], feed_dict={self.input_data:batch_x, self.z:batch_z})

                iteration_time = time.time()
                total_time = (iteration_time - start_time)
                print("epoch[%d]:[%d/%d]: " %(index, idx, batch_idxs), "total_time:{:.4f}".format(total_time), "encoder_loss:{:.4f},decoder_loss:{:.4f},dis_loss:{:.4f}".format(encoder_loss, decoder_loss, dis_loss))

                counter = counter + 1
                if np.mod(idx, 100) == 0:
                    iteration_time = time.time()
                    total_time = (iteration_time - start_time)
                    # sample images and save them
                    samples = self.sess.run(self.sample_images, feed_dict={self.z:sample_z})
                    #print(samples)
                    save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, index, idx))
                    # calc loss
                    sample_encoder_loss, sample_decoder_loss, sample_dis_loss = self.sess.run([self.encoder_loss, self.decoder_loss, self.dis_loss], feed_dict={self.input_data:sample_batch_x, self.z:sample_z})
                                                               
                    print("epoch[%d]:[%d/%d]: " %(index, idx, batch_idxs), "total_time:{:.4f}".format(total_time), "encoder_loss:{:.4f},decoder_loss:{:.4f},dis_loss:{:.4f}".format(sample_encoder_loss, sample_decoder_loss, sample_dis_loss))
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
