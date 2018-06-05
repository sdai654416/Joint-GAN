from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
import pdb
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
from pretrain.utils import  restore_from_save
from tensorflow.contrib.tensorboard.plugins import projector
import cPickle
from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

word, vocab = cPickle.load(open("./pretrain/vocab_cotra.pkl"))

def save_txt(output_file, input_file, vocab = vocab, word = word):
    with open(output_file, 'w') as fout:
        for line in input_file:
            line = [word[x] for x in line]
            while '_PAD' in line: line.remove('_PAD')
            line = ' '.join(line) + '\n'
            fout.write(line)
        
def add_padding(tensor, opt):
    pad_front = opt.filter_shape -1
    pad_back = opt.filter_shape
    paddings = tf.constant([[0, 0,], [pad_front, pad_back]])
    x = tf.pad(tensor, paddings, "CONSTANT")
    return x  

def embedding_view(EMB, y, sess, opt, log_path):
    
    embedding_var = tf.Variable(EMB,  name='Embedding_of_sentence')
    sess.run(embedding_var.initializer)
    EB_summary_writer = tf.summary.FileWriter(log_path)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = os.path.join(log_path, 'metadata.tsv')
    projector.visualize_embeddings(EB_summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(log_path, 'model2.ckpt'), 1)
    metadata_file = open(os.path.join(log_path, 'metadata.tsv'), 'w')
    #metadata_file.write('ClassID\n')
    for i in range(len(y)):
        metadata_file.write('%06d\n' % (y[i]))
    metadata_file.close()
    print('embedding created')
    
def compute_MMD(H_fake, H_real):
    kxx, kxy, kyy = 0, 0, 0
    dividend = 0.1
    dist_x, dist_y = H_fake/dividend, H_real/dividend
    x_sq = np.expand_dims(np.sum(dist_x**2, axis=1), 1)   #  64*1
    y_sq = np.expand_dims(np.sum(dist_y**2, axis=1), 1)    #  64*1
    dist_x_T = np.transpose(dist_x)
    dist_y_T = np.transpose(dist_y)
    x_sq_T = np.transpose(x_sq)
    y_sq_T = np.transpose(y_sq)

    tempxx = -2*np.matmul(dist_x,dist_x_T) + x_sq + x_sq_T  # (xi -xj)**2
    tempxy = -2*np.matmul(dist_x,dist_y_T) + x_sq + y_sq_T  # (xi -yj)**2
    tempyy = -2*np.matmul(dist_y,dist_y_T) + y_sq + y_sq_T  # (yi -yj)**2


    for sigma in [1]:
        kxx += np.mean(np.exp(-tempxx/2/(sigma**2)))
        kxy += np.mean(np.exp(-tempxy/2/(sigma**2)))
        kyy += np.mean(np.exp(-tempyy/2/(sigma**2)))

    #fake_obj = (kxx + kyy - 2*kxy)/n_samples
    #fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)/n_samples
    gan_cost_g = np.sqrt(kxx + kyy - 2*kxy)
    return gan_cost_g

class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="./ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.latent_size = cfg.TRAIN.LATENT_SIZE
        self.n_words = cfg.TRAIN.NUM_WORDS
        self.sent_len = cfg.TRAIN.SENT_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL
        self.opt = cfg.TRAIN.OPTIONS

        self.log_vars = []

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
            
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images')
        
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings')
        
        self.real_words = tf.placeholder(
            tf.int32, [self.batch_size, self.opt.sentence],        
            name='real_words')

        self.real_W_norm = tf.placeholder(
            tf.float32, [self.opt.n_words,self.opt.embed_size],
            name='real_W_norm')
           
        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate')
            
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate')

    
    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0

        return c, cfg.TRAIN.COEFF.KL * kl_loss
    
    
    def init_opt(self):
        self.build_placeholder()

        with pt.defaults_scope(phase=pt.Phase.train):
            
            # #### autoencoder pretrain format ################################
            with tf.variable_scope("g_net"):    
                # real f to Y                
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z2 = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.gen_images = self.model.get_generator(tf.concat([c, z2], 1))
                
                # generate f to Y
                b_zero = tf.zeros([self.batch_size, self.opt.latent_size])                
                self.latent_z1 = tf.random_normal([self.batch_size, self.latent_size])
                self.fake_embeddings = self.model.get_emb_gen(tf.concat([b_zero, self.latent_z1], 1))
                
                c_, _ = self.sample_encoded_context(self.fake_embeddings)
                z2_ = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.gen_fake_images = self.model.get_generator(tf.concat([c_, z2_], 1))
                
                # generate Y to f
                c_zero = tf.zeros([self.batch_size, self.opt.ef_dim])
                self.latent_z2 = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.fake_images = self.model.get_generator(tf.concat([c_zero, self.latent_z2], 1))
                
                b_ = self.model.get_CNN(self.fake_images)
                z1_= tf.random_normal([self.batch_size, self.latent_size])
                self.gen_fake_embeddings = self.model.get_emb_gen(tf.concat([b_, z1_], 1))
                
                # real Y to f
                b = self.model.get_CNN(self.images)
                z1 = tf.random_normal([self.batch_size, self.latent_size])
                self.gen_embeddings = self.model.get_emb_gen(tf.concat([b, z1], 1))
                
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_c_", c_))
                self.log_vars.append(("hist_z", z2))
                self.log_vars.append(("hist_z_", z2_))
                
            with tf.variable_scope("pretrains"):
                _, _, self.fake_words, _ = self.model.get_lstm_decoder_embedding(self.fake_embeddings, self.real_words, 
                                                                                 self.real_W_norm, feed_previous=True)
                                                                                 
            
            # ####get discriminator_loss and generator_loss ###################
            
            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.embeddings,                                    
                                    self.fake_images,
                                    self.fake_embeddings,                                    
                                    self.gen_images,
                                    self.gen_embeddings,
                                    self.gen_fake_images,
                                    self.gen_fake_embeddings,
                                    self.wrong_images
                                    )
            
            # rec_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.latent_z2 - self.rec_latent_z2),1)))
            MMD_b_loss = self.model.get_MMD_loss(b_, b)
            generator_loss += kl_loss + MMD_b_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_MMD_b_loss", MMD_b_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #######Total loss for build optimizers###########################
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("g_net", reuse=True):
                self.sampler_images()
                self.sampler_embeddings()
            with tf.variable_scope("pretrains"):
                _, _, self.gen_fake_words_samples, _ = self.model.get_lstm_decoder_embedding(self.gen_fake_embeddings_samples, self.real_words, 
                                                                                     		 self.real_W_norm, feed_previous=True, is_reuse=True)
            self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler_images(self):
        c, _ = self.sample_encoded_context(self.fake_embeddings)
        if cfg.TRAIN.FLAG:
            z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        else:
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            
        self.gen_fake_images_samples = self.model.get_generator(tf.concat([c, z], 1))
        
    def sampler_embeddings(self):
        b = self.model.get_CNN(self.fake_images)
        if cfg.TRAIN.FLAG:
            z = tf.zeros([self.batch_size, self.latent_size])  # Expect similar BGs
        else:
            z = tf.random_normal([self.batch_size, self.latent_size])
            
        self.gen_fake_embeddings_samples = self.model.get_emb_gen(tf.concat([b, z], 1))
            
    def compute_losses(self, images, embeddings, fake_images, fake_embeddings, 
                       gen_images, gen_embeddings, gen_fake_images, gen_fake_embeddings, 
                       wrong_images):

        l1 = 1
        l2 = 1
        L1_loss = l1 * tf.reduce_mean(tf.abs(gen_images - images))
        L1_loss_ = l1 * tf.reduce_mean(tf.abs(embeddings - gen_embeddings))
        MMD_loss = l2 * self.model.get_MMD_loss(gen_embeddings, embeddings)
        MMD_loss_ = l2 * self.model.get_MMD_loss(gen_fake_embeddings, embeddings)
        
        ###################   Image Parts  ########################
        D1_real = self.model.get_discriminator_1(images, embeddings)
        D2_real = self.model.get_discriminator_2(images, embeddings)
        # D3_real = self.model.get_discriminator_3(images, embeddings)
        D3_real = self.model.get_discriminator_6(images)
        # D4_real = self.model.get_discriminator_4(images, embeddings)
        D4_real = self.model.get_discriminator_5(embeddings)
        wrong_logit = self.model.get_discriminator_1(wrong_images, embeddings)
        D1_fake = self.model.get_discriminator_1(gen_images, embeddings)
        D2_fake = self.model.get_discriminator_2(images, gen_embeddings)
        # D3_fake = self.model.get_discriminator_3(fake_images, gen_fake_embeddings)
        D3_fake = self.model.get_discriminator_6(fake_images)
        # D4_fake = self.model.get_discriminator_4(gen_fake_images, fake_embeddings)
        D4_fake = self.model.get_discriminator_5(fake_embeddings)        
        
        # D1
        real_d1_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_real,
                                                    labels=tf.ones_like(D1_real))
        real_d1_loss = tf.reduce_mean(real_d1_loss)
        
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit,
                                                    labels=tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        
        fake_d1_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake,
                                                    labels=tf.zeros_like(D1_fake))
        fake_d1_loss = tf.reduce_mean(fake_d1_loss) 
        
        # D2
        real_d2_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_real,
                                                    labels=tf.ones_like(D2_real))
        real_d2_loss = tf.reduce_mean(real_d2_loss)
         
        fake_d2_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake,
                                                    labels=tf.zeros_like(D2_fake))
        fake_d2_loss = tf.reduce_mean(fake_d2_loss)
        
        # D3
        real_d3_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D3_real,
                                                    labels=tf.ones_like(D3_real))
        real_d3_loss = tf.reduce_mean(real_d3_loss)
        
        fake_d3_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D3_fake,
                                                    labels=tf.zeros_like(D3_fake))
        fake_d3_loss = tf.reduce_mean(fake_d3_loss)
        
        # D4
        real_d4_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D4_real,
                                                    labels=tf.ones_like(D4_real))
        real_d4_loss = tf.reduce_mean(real_d4_loss)
        
        fake_d4_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D4_fake,
                                                    labels=tf.zeros_like(D4_fake))
        fake_d4_loss = tf.reduce_mean(fake_d4_loss)
        
        # G1
        g1_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_fake,
                                                    labels=tf.ones_like(D1_fake))
        g1_loss = tf.reduce_mean(g1_loss)
        
        # G2
        g2_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_fake,
                                                    labels=tf.ones_like(D2_fake))
        g2_loss = tf.reduce_mean(g2_loss)
        
        # G3
        g3_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D3_fake,
                                                    labels=tf.ones_like(D3_fake))
        g3_loss = tf.reduce_mean(g3_loss)
        
        # G4
        g4_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D4_fake,
                                                    labels=tf.ones_like(D4_fake))
        g4_loss = tf.reduce_mean(g4_loss)
 
        if cfg.TRAIN.B_WRONG:
            discriminator_loss =\
                real_d1_loss + (wrong_d_loss + fake_d1_loss) / 2. + real_d2_loss + fake_d2_loss + real_d3_loss + fake_d3_loss + real_d4_loss + fake_d4_loss
            self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            discriminator_loss = real_d1_loss + fake_d1_loss + real_d2_loss + fake_d2_loss + real_d3_loss + fake_d3_loss + real_d4_loss + fake_d4_loss
            
        generator_loss = g1_loss + g2_loss + g3_loss + g4_loss + MMD_loss + MMD_loss_ + L1_loss + L1_loss_
        
        self.log_vars.append(("d_1_loss_real", real_d1_loss))
        self.log_vars.append(("d_1_loss_fake", fake_d1_loss))
        self.log_vars.append(("d_2_loss_real", real_d2_loss))
        self.log_vars.append(("d_2_loss_fake", fake_d2_loss))
        self.log_vars.append(("d_3_loss_real", real_d3_loss))
        self.log_vars.append(("d_3_loss_fake", fake_d3_loss))
        self.log_vars.append(("d_4_loss_real", real_d4_loss))
        self.log_vars.append(("d_4_loss_fake", fake_d4_loss))
        self.log_vars.append(("g_1_loss", g1_loss))
        self.log_vars.append(("g_2_loss", g2_loss))
        self.log_vars.append(("g_3_loss", g3_loss))
        self.log_vars.append(("g_4_loss", g4_loss))
        self.log_vars.append(("g_L1_loss", L1_loss))
        self.log_vars.append(("g_L1_loss_", L1_loss_))
        self.log_vars.append(("g_MMD_loss", MMD_loss))
        self.log_vars.append(("g_MMD_loss_", MMD_loss_))
        return discriminator_loss, generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if 
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]
        pretrain_vars = [var for var in all_vars if
                         var.name.startswith('pretrain')]
        
                
        
        generator_opt = tf.train.AdamOptimizer(self.generator_lr,
                                               beta1=0.5)
        self.generator_trainer =\
            pt.apply_optimizer(generator_opt,
                               losses=[generator_loss],
                               var_list=g_vars)
        discriminator_opt = tf.train.AdamOptimizer(self.discriminator_lr,
                                                   beta1=0.5)
        self.discriminator_trainer =\
            pt.apply_optimizer(discriminator_opt,
                               losses=[discriminator_loss],
                               var_list=d_vars)
                               
        self.log_vars.append(("g_learning_rate", self.generator_lr))
        self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.summary.histogram(k, v))

        self.g_sum = tf.summary.merge(all_sum['g'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        self.hist_sum = tf.summary.merge(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.expand_dims(tf.concat(stacked_img, 0), 0)
        current_img_summary = tf.summary.image(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        # real
        gen_fake_sum, superimage_gen_fake = \
            self.visualize_one_superimage(self.gen_fake_images_samples[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum, superimage_fake = \
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "test")
        
        self.superimages = tf.concat([superimage_gen_fake, superimage_fake], 0)
        self.image_summary = tf.summary.merge([gen_fake_sum, fake_sum])
        

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):
        fix_real_word = np.loadtxt('./pretrain/real_cotra.txt', dtype=int)
        fix_real_word = fix_real_word[:self.opt.batch_size]
        W_norm = np.load('./pretrain/embeddings_4391.npy')        
        
        
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ = \
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        feed_dict = {self.images: images,
                     self.embeddings: embeddings,
                     self.real_words: fix_real_word,
                     self.real_W_norm: W_norm}
        gen_samples, img_summary, fake_word, gen_fake_word =\
            sess.run([self.superimages, self.image_summary, self.fake_words, self.gen_fake_words_samples], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/gen_fake_images.jpg' % (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/fake_images.jpg' % (self.log_dir), gen_samples[1])
        
        save_txt(self.checkpoint_dir+'/fake_sentences.txt', fake_word)
        save_txt(self.checkpoint_dir+'/gen_fake_sentences.txt', gen_fake_word)
        # np.savetxt(self.checkpoint_dir+'/fake_words.txt', fake_word, fmt='%i', delimiter=' ')
        # np.savetxt(self.checkpoint_dir+'/gen_fake_words.txt', gen_fake_word, fmt='%i', delimiter=' ')
        
        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.initialize_all_variables())
        all_var = tf.trainable_variables() 
        
        try:
            t_vars = tf.trainable_variables()
            loader = restore_from_save(t_vars, sess, self.opt)
            print('Load pretrain successfully')
        except Exception as e:
            print(e)
            print("No saving session, using random initialization")
            sess.run(tf.global_variables_initializer())        
        
        
        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            restore_vars = tf.all_variables()
            # all_vars = tf.all_variables()
            # restore_vars = [var for var in all_vars if not
            #                 var.name.startswith('pretrain')]
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train(self):       
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        with tf.Session(config=config) as sess:            
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(),
                                       keep_checkpoint_every_n_hours=2)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)

                keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                        # print(k, v)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                num_embedding = cfg.TRAIN.NUM_EMBEDDING
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch != 0:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        # training d
                        images, wrong_images, embeddings, _, _ =\
                            self.dataset.train.next_batch(self.batch_size,
                                                          num_embedding)
                        
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_lr: generator_lr,
                                     self.discriminator_lr: discriminator_lr
                                     }
                        # train d
                        feed_out = [self.discriminator_trainer,
                                    self.d_sum,
                                    self.hist_sum,
                                    log_vars,
                                    self.embeddings,
                                    self.fake_embeddings]
                        
                        for j in range(self.opt.dis_steps):
                            _, d_sum, hist_sum, log_vals, real_emb, fake_emb = sess.run(feed_out,
                                                                                        feed_dict)
                        
                        
                        summary_writer.add_summary(d_sum, counter)
                        summary_writer.add_summary(hist_sum, counter)
                        all_log_vals.append(log_vals)

                        # train g
                        feed_out = [self.generator_trainer,
                                    self.g_sum]
                        for k in range(self.opt.gen_steps):
                            _, g_sum = sess.run(feed_out,
                                                feed_dict)
                                                
                        summary_writer.add_summary(g_sum, counter)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            
                            EMB = np.concatenate((real_emb, fake_emb))
                            y = np.zeros(EMB.shape[0])
                            y[:real_emb.shape[0]] = 1
                                                        
                            
                            print("Model saved in file: %s" % fn)

                    img_sum = self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_sum, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

    def save_super_images(self, images, sample_batchs, filenames,
                          sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        numSamples = len(sample_batchs)
        for j in range(len(filenames)):
            s_tmp = '%s-1real-%dsamples/%s/%s' %\
                (save_dir, numSamples, subset, filenames[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            superimage = [images[j]]
            # cfg.TRAIN.NUM_COPY samples for each text embedding/sentence
            for i in range(len(sample_batchs)):
                superimage.append(sample_batchs[i][j])

            superimage = np.concatenate(superimage, axis=1)
            fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
            scipy.misc.imsave(fullpath, superimage)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images, embeddings_batchs, filenames, _ =\
                dataset.next_batch_test(self.batch_size, start, 1)
            print('count = ', count, 'start = ', start)
            for i in range(len(embeddings_batchs)):
                samples_batchs = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                for j in range(np.minimum(16, cfg.TRAIN.NUM_COPY)):
                    samples = sess.run(self.fake_images,
                                       {self.embeddings: embeddings_batchs[i]})
                    samples_batchs.append(samples)
                self.save_super_images(images, samples_batchs,
                                       filenames, i, save_dir,
                                       subset)

            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.all_variables())
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")
