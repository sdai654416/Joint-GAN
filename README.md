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
1. Download the pretrained LSTM decoder for [bird](https://drive.google.com/open?id=1j9do5K1BbghwD6W--XvJmbhj21XEEqjV) save all files to `pretrain/`

**Training**

Train a Joint GAN model on the CUB dataset using the preprocessed data for birds: `python Main.py`
