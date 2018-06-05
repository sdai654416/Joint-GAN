from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.ops import nn_ops, math_ops
from tensorflow.contrib.legacy_seq2seq import sequence_loss, embedding_rnn_decoder
from tensorflow.contrib import layers
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg
from math import floor

from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import nn_ops, math_ops, embedding_ops, variable_scope, array_ops
# from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
# from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
# from tensorflow.contrib.seq2seq.python.ops import decoder

import pretrain.data_utils as dp

def regularization(X, batch_norm, is_train, prefix= '', is_reuse= None, dropout = False):
    if '_X' not in prefix and '_H_dec' not in prefix:
        if batch_norm:
            X = layers.batch_norm(X, decay=0.9, center=True, scale=True, is_training=is_train, scope=prefix+'_bn', reuse = is_reuse)
        X = tf.nn.tanh(X)
    X = X if (not dropout or is_train is None) else layers.dropout(X, keep_prob = 0.5, scope=prefix + '_dropout')
    return X
    
def normalizing(x, axis):    
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
    normalized = x / (norm)   
    return normalized


class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.emb_dim = cfg.GAN.EMBEDDING_SIZE
        self.n = 1
        # Word Options        
        self.opt = cfg.TRAIN.OPTIONS
        with tf.variable_scope('g_'):
            self.CNN_img_template = self.CNN()
        
        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template_1 = self.d_encode_image_1()
                self.d_context_template_1 = self.context_embedding_1()
                self.discriminator_template_1 = self.discriminator_1()
                
                self.d_encode_img_template_2 = self.d_encode_image_2()
                self.d_context_template_2 = self.context_embedding_2()
                self.discriminator_template_2 = self.discriminator_2()
                
                self.d_encode_img_template_3 = self.d_encode_image_3()
                self.d_context_template_3 = self.context_embedding_3()
                self.discriminator_template_3 = self.discriminator_3()
                
                self.d_encode_img_template_4 = self.d_encode_image_4()
                self.d_context_template_4 = self.context_embedding_4()
                self.discriminator_template_4 = self.discriminator_4()
                
                self.d_encode_img_template_6 = self.d_encode_image_6()
                self.discriminator_template_6 = self.discriminator_6()
                
                
        elif cfg.GAN.NETWORK_TYPE == "simple":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image_simple()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    def generate_condition(self, c_var):
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             # apply(leaky_rectify, leakiness=0.2))
             apply(tf.nn.sigmoid))
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    def emb_gen(self, H, opt, prefix = '', is_reuse= None):
        # last layer must be linear
        H = tf.squeeze(H)
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        H_1 = layers.fully_connected(tf.nn.dropout(H, keep_prob = opt.dropout_ratio), 
                                       num_outputs = opt.H_dis, biases_initializer=biasInit, 
                                       activation_fn = tf.nn.relu, scope = prefix + 'emb_gen_1', reuse = is_reuse)
                                       
        H_2 = layers.fully_connected(tf.nn.dropout(H_1, keep_prob = opt.dropout_ratio), 
                                       num_outputs = opt.H_dis, biases_initializer=biasInit, 
                                       activation_fn = tf.nn.relu, scope = prefix + 'emb_gen_2', reuse = is_reuse)
                                       
        logits = 2*layers.linear(tf.nn.dropout(H_2, keep_prob = opt.dropout_ratio), 
                                 num_outputs = opt.n_gan, biases_initializer=biasInit, 
                                 activation_fn = tf.nn.tanh, scope = prefix + 'emb_gen_3', reuse = is_reuse)

        return logits
    
    def get_emb_gen(self, H):
        return self.emb_gen(H, self.opt)

    def generator(self, z_var):
        node1_0 =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]))

        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def generator_simple(self, z_var):
        output_tensor =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             # custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             # custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             # custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             # custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def get_generator(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var)
        else:
            raise NotImplementedError
    
    # original backup
    def generator_2(self, z_var):
        node1_0 =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor
        
    def get_generator_2(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator_2(z_var)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var)
        else:
            raise NotImplementedError  

    # d-net
    def context_embedding_1(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template
        
    def context_embedding_2(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template
    
    def context_embedding_3(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template
    
    def context_embedding_4(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def d_encode_image_1(self):
        n=self.n
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim//n, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim//n * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim//n * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1
        
    def d_encode_image_2(self):
        n=self.n
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim//n, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim//n * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim//n * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))
             
        return node1
        
    def d_encode_image_3(self):
        n=self.n
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim//n, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim//n * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim//n * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))
             
        return node1
        
    def d_encode_image_4(self):
        n=self.n
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim//n, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim//n * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim//n * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
             
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))
        
        return node1

    def d_encode_image_simple(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2))

        return template

    def discriminator_1(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template
    
    def discriminator_2(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template
        
    def discriminator_3(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template
        
    def discriminator_4(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template
        
    def get_discriminator_1(self, x_var, c_var):
        x_code = self.d_encode_img_template_1.construct(input=x_var)

        c_code = self.d_context_template_1.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])

        x_c_code = tf.concat([x_code, c_code], 3)
        return self.discriminator_template_1.construct(input=x_c_code)
        
    def get_discriminator_2(self, x_var, c_var):
        x_code = self.d_encode_img_template_2.construct(input=x_var)

        c_code = self.d_context_template_2.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])

        x_c_code = tf.concat([x_code, c_code], 3)
        return self.discriminator_template_2.construct(input=x_c_code)
        
    def get_discriminator_3(self, x_var, c_var):
        x_code = self.d_encode_img_template_3.construct(input=x_var)

        c_code = self.d_context_template_3.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])

        x_c_code = tf.concat([x_code, c_code], 3)
        return self.discriminator_template_3.construct(input=x_c_code)
        
    def get_discriminator_4(self, x_var, c_var):
        x_code = self.d_encode_img_template_4.construct(input=x_var)

        c_code = self.d_context_template_4.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])

        x_c_code = tf.concat([x_code, c_code], 3)
        return self.discriminator_template_4.construct(input=x_c_code)
        
    def discriminator_5(self, H, opt, prefix = 'd_net', is_reuse= None, is_train= True):
        # last layer must be linear
        H = tf.squeeze(H)
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        H = regularization(H, opt, is_train, prefix= prefix + '/reg_H', is_reuse= is_reuse)
        H_dis = layers.fully_connected(H, num_outputs = opt.H_dis, biases_initializer=biasInit, activation_fn = tf.nn.relu, scope = prefix + 'dis_1', reuse = is_reuse)
        H_dis = regularization(H_dis, opt, is_train, prefix= prefix + '/reg_H_dis', is_reuse= is_reuse)
        logits = layers.linear(H_dis, num_outputs = 1, biases_initializer=biasInit, scope = prefix + '/disc', reuse = is_reuse)
        return logits
    
    def get_discriminator_5(self, H, is_reuse= None):
        return self.discriminator_5(H, self.opt, is_reuse=is_reuse)
        
    def d_encode_image_6(self):
        n=self.n
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim//n, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim//n * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim//n * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1
        
    def discriminator_6(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))
        
        return template
        
    def get_discriminator_6(self, x_var):
        x_code = self.d_encode_img_template_6.construct(input=x_var)
        output= self.discriminator_template_6.construct(input=x_code)
        return output
    
    def deconv_model_3layer(self, H, opt, prefix = '', is_reuse= None, is_train = True, multiplier = 2):
        #XX = tf.reshape(X, [-1, , 28, 1])
        #X shape: batchsize L emb 1
        biasInit = None if opt.batch_norm else tf.constant_initializer(0.001, dtype=tf.float32)
    
        H3t = H
        
        H3t = regularization(H3t, opt, prefix= prefix + 'reg_H_dec', is_reuse= is_reuse, is_train = is_train)
        H2t = layers.conv2d_transpose(H3t, num_outputs=opt.filter_size*multiplier,  kernel_size=[opt.sent_len3, 1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H2_t_3', reuse = is_reuse)
    
        H2t = regularization(H2t, opt, prefix= prefix + 'reg_H2_dec', is_reuse= is_reuse, is_train = is_train)
        H1t = layers.conv2d_transpose(H2t, num_outputs=opt.filter_size,  kernel_size=[opt.filter_shape, 1], stride = [opt.stride[1],1],  biases_initializer=biasInit, activation_fn=None ,padding = 'VALID', scope = prefix + 'H1_t_3', reuse = is_reuse)
    
        H1t = regularization(H1t, opt, prefix= prefix + 'reg_H1_dec', is_reuse= is_reuse, is_train = is_train)
        Xhat = layers.conv2d_transpose(H1t, num_outputs=1,  kernel_size=[opt.filter_shape, opt.embed_size], stride = [opt.stride[0],1],  biases_initializer=None, activation_fn=tf.nn.relu, padding = 'VALID',scope = prefix + 'Xhat_t_3', reuse = is_reuse)
        #print H2t.get_shape(),H1t.get_shape(),Xhat.get_shape()
    
        return Xhat
    
    
    def deconv_decoder(self, H_dec, x_org, W_norm, opt, res = {}, prefix = '', is_reuse = None):
        x_rec = self.deconv_model_3layer(H_dec, opt)  #  batch L emb 1
        
        print("Decoder len %d Output len %d" % (x_rec.get_shape()[1], x_org.get_shape()[1]))
        tf.assert_equal(x_rec.get_shape()[1], x_org.get_shape()[1])
        x_rec_norm = normalizing(x_rec, 2)    # batch L emb
        x_temp = tf.reshape(x_org, [-1,])
        if hasattr(opt, 'attentive_emb') and opt.attentive_emb:
            emb_att = tf.get_variable(prefix+'emb_att', [1,opt.embed_size], initializer = tf.constant_initializer(1.0, dtype=tf.float32))
            prob_logits = tf.tensordot(tf.squeeze(x_rec_norm), emb_att*W_norm, [[2],[1]])  # c_blv = sum_e x_ble W_ve
        else:
            prob_logits = tf.tensordot(tf.squeeze(x_rec_norm), W_norm, [[2],[1]])  # c_blv = sum_e x_ble W_ve
    
        prob = tf.nn.log_softmax(prob_logits*opt.L, dim=-1, name=None)
        #prob = normalizing(tf.reduce_sum(x_rec_norm * W_reshape, 2), 2)
        #prob = softmax_prediction(x_rec_norm, opt)
        rec_sent = tf.squeeze(tf.argmax(prob,2))
        prob = tf.reshape(prob, [-1,opt.n_words])
    
        idx = tf.range(opt.batch_size * opt.sent_len)
        #print idx.get_shape(), idx.dtype
    
        all_idx = tf.transpose(tf.stack(values=[idx,x_temp]))
        all_prob = tf.gather_nd(prob, all_idx)

    
        gen_temp = tf.cast(tf.reshape(rec_sent, [-1,]), tf.int32)
        gen_idx = tf.transpose(tf.stack(values=[idx,gen_temp]))
        gen_prob = tf.gather_nd(prob, gen_idx)
    
        res['rec_sent'] = rec_sent

        if opt.discrimination:
            logits_real, _ = discriminator(x_org, W_norm, opt)
            prob_one_hot = tf.nn.log_softmax(prob_logits*opt.L, dim=-1, name=None)
            logits_syn, _ = discriminator(tf.exp(prob_one_hot), W_norm, opt, is_prob = True, is_reuse = True)
    
            res['prob_r'] =  tf.reduce_mean(tf.nn.sigmoid(logits_real))
            res['prob_f'] = tf.reduce_mean(tf.nn.sigmoid(logits_syn))
    
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits = logits_real)) + \
                         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_syn), logits = logits_syn))
        else:
            loss = -tf.reduce_mean( all_prob)
        return loss, res['rec_sent']
        
    def get_deconv_decoder(self, H_dec, x_org, W_norm):
        return self.deconv_decoder(H_dec, x_org, W_norm, self.opt)
    
    
    
    def embedding_only(self, opt, prefix = '', is_reuse = None):
        """Customized function to transform batched x into embeddings."""
        # Convert indexes of words into embeddings.
        with tf.variable_scope(prefix+'embed', reuse=is_reuse):
            if opt.fix_emb:
                assert(hasattr(opt,'emb'))
                assert(np.shape(np.array(opt.emb))==(opt.n_words, opt.embed_size))
                W = tf.get_variable('W', [opt.n_words, opt.embed_size], weights_initializer = opt.emb, is_trainable = False)
            else:
                weightInit = tf.random_uniform_initializer(-0.001, 0.001)
                W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer = weightInit)
        #    b = tf.get_variable('b', [opt.embed_size], initializer = tf,random_uniform_initializer(-0.01, 0.01))
        if hasattr(opt, 'relu_w') and opt.relu_w:
            W = tf.nn.relu(W)
    
        W_norm = normalizing(W, 1)
    
        return W_norm 
        
    def get_embedding_only(self):
        return self.embedding_only(self.opt)
    
    
    
    def embedding(self, features, opt, prefix = '', is_reuse = None):
        """Customized function to transform batched x into embeddings."""
        with tf.variable_scope(prefix+'embed', reuse=is_reuse):
            if opt.fix_emb:
                assert(hasattr(opt,'emb'))
                assert(np.shape(np.array(opt.emb))==(opt.n_words, opt.embed_size))
                W = tf.get_variable('W', [opt.n_words, opt.embed_size], weights_initializer = opt.emb, is_trainable = False)
            else:
                weightInit = tf.random_uniform_initializer(-0.001, 0.001)
                W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer = weightInit)
            # tf.stop_gradient(W)
        if hasattr(opt, 'relu_w') and opt.relu_w:
            W = tf.nn.relu(W)
    
        W_norm = normalizing(W, 1)
        word_vectors = tf.nn.embedding_lookup(W_norm, features)
    
    
        return word_vectors, W_norm
    
    def get_embedding(self, features):
        return self.embedding(features, self.opt)
        
    def CNN(self):
        n=self.n
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim//n, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim//n * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim//n * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim//n * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2).
             flatten().
             custom_fully_connected(self.opt.latent_size * 16).
             apply(leaky_rectify, leakiness=0.2).
             fc_batch_norm().
             custom_fully_connected(self.opt.latent_size).
             apply(tf.nn.tanh))
             
        return node1
        
    def get_CNN(self, X):
        
        output = self.CNN_img_template.construct(input=X)
        return tf.convert_to_tensor(output)
        
        
    def classifier_2layer(self, H, opt, prefix = '', is_reuse= None):
        # last layer must be linear
        H = tf.squeeze(H)
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob = opt.dropout_ratio), 
                                       num_outputs = opt.H_dis, biases_initializer=biasInit, 
                                       activation_fn = tf.nn.relu, scope = prefix + 'class_1', reuse = is_reuse)
        logits = layers.linear(tf.nn.dropout(H_dis, keep_prob = opt.dropout_ratio), 
                               num_outputs = opt.ef_dim, biases_initializer=biasInit, 
                               activation_fn = tf.nn.tanh, scope = prefix + 'class_2', reuse = is_reuse)
        return logits

    def get_classifier_2layer(self, H):
        return self.classifier_2layer(H, self.opt)
        
    def vae_classifier_2layer(self, H, opt, prefix = '', is_reuse= None):
        # last layer must be linear
        H = tf.squeeze(H)
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob = opt.dropout_ratio), 
                                       num_outputs = opt.n_gan, biases_initializer=biasInit, 
                                       activation_fn = tf.nn.relu, scope = prefix + 'vae_1', reuse = is_reuse)
        mean = layers.linear(tf.nn.dropout(H_dis, keep_prob = opt.dropout_ratio), 
                                   num_outputs = opt.n_gan, biases_initializer=biasInit, 
                                   activation_fn = None, scope = prefix + 'vae_2', reuse = is_reuse)
        log_sigma_sq = layers.linear(tf.nn.dropout(H_dis, keep_prob = opt.dropout_ratio), 
                                   num_outputs = opt.n_gan, biases_initializer=biasInit, 
                                   activation_fn = None, scope = prefix + 'vae_3', reuse = is_reuse)                           
        
        return mean, log_sigma_sq
        
    def get_vae_classifier_2layer(self, H):
        return self.vae_classifier_2layer(H, self.opt)


    def compute_MMD_loss(self, H_fake, H_real, opt):
        kxx, kxy, kyy = 0, 0, 0
        dividend = 1
        dist_x, dist_y = H_fake/dividend, H_real/dividend
        x_sq = tf.expand_dims(tf.reduce_sum(dist_x**2,axis=1), 1)   #  64*1
        y_sq = tf.expand_dims(tf.reduce_sum(dist_y**2,axis=1), 1)    #  64*1
        dist_x_T = tf.transpose(dist_x)
        dist_y_T = tf.transpose(dist_y)
        x_sq_T = tf.transpose(x_sq)
        y_sq_T = tf.transpose(y_sq)

        tempxx = -2*tf.matmul(dist_x,dist_x_T) + x_sq + x_sq_T  # (xi -xj)**2
        tempxy = -2*tf.matmul(dist_x,dist_y_T) + x_sq + y_sq_T  # (xi -yj)**2
        tempyy = -2*tf.matmul(dist_y,dist_y_T) + y_sq + y_sq_T  # (yi -yj)**2


        for sigma in opt.sigma_range:
            kxx += tf.reduce_mean(tf.exp(-tempxx/2/(sigma**2)))
            kxy += tf.reduce_mean(tf.exp(-tempxy/2/(sigma**2)))
            kyy += tf.reduce_mean(tf.exp(-tempyy/2/(sigma**2)))

        #fake_obj = (kxx + kyy - 2*kxy)/n_samples
        #fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)/n_samples
        gan_cost_g = tf.sqrt(kxx + kyy - 2*kxy)
        return gan_cost_g
        
    def get_MMD_loss(self, H_fake, H_real):
        return self.compute_MMD_loss(H_fake, H_real, self.opt)
    
    
    def lstm_decoder_embedding(self, H, y, W_emb, opt, prefix = '', add_go = True, feed_previous=False, is_reuse= None, is_fed_h = True, is_sampling = False, is_softargmax = False, beam_width=None):
        #y  len* batch * [0,V]   H batch * h
        biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
        #y = [tf.squeeze(y[:,i]) for i in xrange(y.get_shape()[1])]
        if add_go:
            y = tf.concat([tf.ones([opt.batch_size,1],dtype=tf.int32), y],1)
    
        y = tf.unstack(y, axis=1)  # 1, . , .
        # make the size of hidden unit to be n_hid
        if not opt.additive_noise_lambda:
            H = layers.fully_connected(H, num_outputs = opt.n_hid, biases_initializer=biasInit, activation_fn = None, scope = prefix + 'lstm_decoder', reuse = is_reuse)
        H0 = tf.squeeze(H)
        H1 = (H0, tf.zeros_like(H0))  # initialize H and C #
    
        y_input = [tf.concat([tf.nn.embedding_lookup(W_emb, features),H0],1) for features in y] if is_fed_h   \
                   else [tf.nn.embedding_lookup(W_emb, features) for features in y]
        with tf.variable_scope(prefix + 'lstm_decoder', reuse=True):
            cell = tf.contrib.rnn.LSTMCell(opt.n_hid)
        with tf.variable_scope(prefix + 'lstm_decoder', reuse=is_reuse):
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_hid, opt.embed_size], initializer = weightInit)
            b = tf.get_variable('b', [opt.n_words], initializer = tf.random_uniform_initializer(-0.001, 0.001))
            W_new = tf.matmul(W, W_emb, transpose_b=True) # h* V
    
            out_proj = (W_new,b) if feed_previous else None
            decoder_res = self.rnn_decoder_custom_embedding(emb_inp = y_input, initial_state = H1, cell = cell, embedding = W_emb, opt = opt, feed_previous = feed_previous, output_projection=out_proj, num_symbols = opt.n_words, is_fed_h = is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling)
            outputs = decoder_res[0]
    
            if beam_width:
                #cell = rnn_cell.LSTMCell(cell_depth)
                #batch_size_tensor = constant_op.constant(opt.batch_size)
                initial_state = cell.zero_state(opt.batch_size* beam_width, tf.float32) #beam_search_decoder.tile_batch(H0, multiplier=beam_width)
                output_layer = layers_core.Dense(opt.n_words, use_bias=True, kernel_initializer = W_new, bias_initializer = b, activation=None)
                bsd = beam_search_decoder.BeamSearchDecoder(
                    cell=cell,
                    embedding=W_emb,
                    start_tokens=array_ops.fill([opt.batch_size], dp.GO_ID), # go is 1
                    end_token=dp.EOS_ID,
                    initial_state=initial_state,
                    beam_width=beam_width,
                    output_layer=output_layer,
                    length_penalty_weight=0.0)
                #pdb.set_trace()
                final_outputs, final_state, final_sequence_lengths = (
                    decoder.dynamic_decode(bsd, output_time_major=False, maximum_iterations=opt.maxlen))
                beam_search_decoder_output = final_outputs.beam_search_decoder_output
                #print beam_search_decoder_output.get_shape()
    
        logits = [nn_ops.xw_plus_b(out, W_new, b) for out in outputs]  # hidden units to prob logits: out B*h  W: h*E  Wemb V*E
        if is_sampling:
            syn_sents = decoder_res[2]
            loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.ones_like(yy),tf.float32) for yy in syn_sents])
            #loss = sequence_loss(logits[:-1], syn_sents, [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in syn_sents])
            #loss = sequence_loss(logits[:-1], syn_sents, [tf.concat([tf.ones([1]), tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32)],0) for yy in syn_sents[:-1]]) # use one more pad after EOS
            syn_sents = tf.stack(syn_sents,1)
        else:
            syn_sents = [math_ops.argmax(l, 1) for l in logits]
            syn_sents = tf.stack(syn_sents,1)
            loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.ones_like(yy),tf.float32) for yy in y[1:]])
            prob = [tf.nn.softmax(l*opt.L) for l in logits]
            prob = tf.stack(prob,1)  # B L V??
            #loss = sequence_loss(logits[:-1], y[1:], [tf.cast(tf.not_equal(yy,dp.PAD_ID),tf.float32) for yy in y[:-1]]) # use one more pad after EOS
    
        #outputs, _ = embedding_rnn_decoder(decoder_inputs = y, initial_state = H, cell = tf.contrib.rnn.BasicLSTMCell, num_symbols = opt.n_words, embedding_size = opt.embed_size, scope = prefix + 'lstm_decoder')
    
        # outputs : batch * len

        return loss, prob, syn_sents, logits

    
    
    def get_lstm_decoder_embedding(self, H, y, W_emb, feed_previous = False, is_reuse=None):
        return self.lstm_decoder_embedding(H, y, W_emb, self.opt, feed_previous = feed_previous, is_reuse=is_reuse)
    
    def rnn_decoder_custom_embedding(self, 
                                     emb_inp,
                                     initial_state,
                                     cell,
                                     embedding,
                                     opt,
                                     num_symbols,
                                     output_projection=None,
                                     feed_previous=False,
                                     update_embedding_for_previous=True,
                                     scope=None,
                                     is_fed_h = True,
                                     is_softargmax = False,
                                     is_sampling = False
                                     ):
    
      with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
        if output_projection is not None:
          dtype = scope.dtype
          proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
          proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
          proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
          proj_biases.get_shape().assert_is_compatible_with([num_symbols])
    
        # embedding = variable_scope.get_variable("embedding",
        #                                         [num_symbols, embedding_size])
        loop_function = self._extract_argmax_and_embed(
            embedding, initial_state[0], opt, output_projection,
            update_embedding_for_previous, is_fed_h=is_fed_h, is_softargmax = is_softargmax, is_sampling = is_sampling) if feed_previous else None
    
        custom_decoder = self.rnn_decoder_with_sample if is_sampling else self.rnn_decoder_truncated
    
        return custom_decoder(emb_inp, initial_state, cell, loop_function=loop_function, truncate = opt.bp_truncation)
    
    
    def _extract_argmax_and_embed(self, 
                                  embedding,
                                  h,
                                  opt,
                                  output_projection=None,
                                  update_embedding=True,
                                  is_fed_h = True,
                                  is_softargmax = False,
                                  is_sampling = False):
    
      def loop_function_with_sample(prev, _):
        if output_projection is not None:
          prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        if is_sampling:
          prev_symbol_sample = tf.squeeze(tf.multinomial(prev*opt.L,1))  #B 1   multinomial(log odds)
          prev_symbol_sample = array_ops.stop_gradient(prev_symbol_sample) # important
          emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol_sample)
        else:
          if is_softargmax:
            prev_symbol_one_hot = tf.nn.log_softmax(prev*opt.L)  #B V
            emb_prev = tf.matmul( tf.exp(prev_symbol_one_hot), embedding) # solve : Requires start <= limit when delta > 0
          else:
            prev_symbol = math_ops.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        emb_prev = tf.concat([emb_prev,h], 1) if is_fed_h else emb_prev
        if not update_embedding: #just update projection?
          emb_prev = array_ops.stop_gradient(emb_prev)
        return (emb_prev, prev_symbol_sample) if is_sampling else emb_prev
    
      # def loop_function(prev, _):
      #   if is_sampling:
      #     emb_prev, _ = loop_function_with_sample(prev, _)
      #   else:
      #     emb_prev = loop_function_with_sample(prev, _)
      #   return emb_prev
    
      return loop_function_with_sample #if is_sampling else loop_function
    
    
    def rnn_decoder_truncated(self,
                    decoder_inputs,
                    initial_state,
                    cell,
                    loop_function=None,
                    scope=None,
                    truncate=None):
      with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
          if loop_function is not None and prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
              inp = loop_function(prev, i)
          if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
          output, state = cell(inp, state)
          if i >0 and truncate and tf.mod(i,truncate) == 0:
            #tf.stop_gradient(state)
            tf.stop_gradient(output)
          outputs.append(output)
          if loop_function is not None:
            prev = output
      return outputs, state
    
    
    def rnn_decoder_with_sample(self,
                    decoder_inputs,
                    initial_state,
                    cell,
                    loop_function=None,
                    scope=None,
                    truncate=None):
      with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs, sample_sent = [], []
        prev = None
        for i, inp in enumerate(decoder_inputs):
          if loop_function is not None and prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
              inp, cur_token = loop_function(prev, i)
            sample_sent.append(cur_token)
          if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
          output, state = cell(inp, state)
          if i >0 and truncate and tf.mod(i,truncate) == 0:
            #tf.stop_gradient(state)
            tf.stop_gradient(output)
          outputs.append(output)
          if loop_function is not None:
            prev = output
      return outputs, state, sample_sent
        
    
