# Joint-GAN

Tensorflow implementation for reproducing results in Joint GAN. Implemented based on [StackGAN](https://github.com/hanzhanggit/StackGAN).

### Dependencies
- `python 2.7`
- `TensorFlow 1.0.0`
- `prettytensor`
- `progressbar`
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`


**Data**
1. Download the preprocessed char-CNN-RNN text embeddings for [birds](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE) and save them to `Data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data and extract to `Data/birds/`
3. Preprocess images: `python misc/preprocess_birds.py`

**Pretrained Model**

Download the pretrained LSTM decoder for [bird](https://drive.google.com/open?id=1j9do5K1BbghwD6W--XvJmbhj21XEEqjV) and unzip all files to `pretrain/`

**Training**

Train a Joint GAN model on the CUB dataset using the preprocessed data for birds: `python Main.py`

**Results**

Generated results can be find in `ckt_logs/birds/`
- fake_images: generated images from noise
- gen_fake_sentences: conditionally generated sentences based on fake_images
- fake_sentences: generated sentences from noise
- gen_fake_images: conditionally generated images based on fake_sentences

The very left column of each image file is sample real images. The rest 16 images are paired with the first 16 sentences in the corresponding text file. 
