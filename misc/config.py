from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from math import floor

__C = edict()
cfg = __C

class Options(object):
    def __init__(self):
        self.dis_steps = 1
        self.gen_steps = 1
	
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' #'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 41
        self.n_words = 4391
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2
        self.embed_size = 620
        self.latent_size = 100
        self.lr = 1e-4

        self.rnn_share_emb = True
        self.additive_noise_lambda = 0.0
        self.bp_truncation = None
        self.n_hid = 100

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 64
        self.max_epochs = 100
        self.n_gan = 1024  # self.filter_size * 3
        self.L = 100

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None #None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.relu_w = False
	
        self.save_path = "./pretrain/save/" + "bird_" + str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(self.permutation)
        self.log_path = "./pretrain/log"
        self.print_freq = 1000
        self.valid_freq = 1000

        # batch norm & dropout
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 0.5

        self.discrimination = False
        self.H_dis = 300
        self.ef_dim = 128
        self.sigma_range = [5]

        self.sent_len = 40
        self.sentence = self.maxlen - 1
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = 1
__C.Z_DIM = 128

# Demo/test options
__C.TEST = edict()
__C.TEST.LR_IMSIZE = 64
__C.TEST.HR_IMSIZE = 256
__C.TEST.NUM_COPY = 16
__C.TEST.BATCH_SIZE = 64
__C.TEST.NUM_COPY = 16
__C.TEST.PRETRAINED_MODEL = ''
__C.TEST.CAPTION_PATH = ''


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.LATENT_SIZE = 100
__C.TRAIN.NUM_WORDS = 4391 #size of the dictionary
__C.TRAIN.SENT_LEN = 40
__C.TRAIN.NUM_COPY = 4
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 1000
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600


__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.LR_DECAY_EPOCH = 50

__C.TRAIN.NUM_EMBEDDING = 4
__C.TRAIN.COND_AUGMENTATION = True
__C.TRAIN.B_WRONG = True

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

__C.TRAIN.FINETUNE_LR = False
__C.TRAIN.FT_LR_RETIO = 0.1

# Word Options
__C.TRAIN.OPTIONS = Options()

# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 128
__C.GAN.EMBEDDING_SIZE = 1024
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.NETWORK_TYPE = 'default'


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
