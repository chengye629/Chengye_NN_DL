
# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = 'D:/NN_DL/PJ1/codes/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = 'D:/NN_DL/PJ1/codes/dataset/MNIST/train-labels-idx1-ubyte.gz'
# test_images_path = 'D:/NN_DL/PJ1/dataset/MNIST/t10k-images-idx3-ubyte.gz'
# test_labels_path = 'D:/NN_DL/PJ1/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
# linear_model_neo = nn.models.Model_MLP_NEO(train_imgs.shape[-1], [512,128], 10, 'ReLU', [1e-4, 1e-4, 1e-4],dropout_rates=[0.2, 0.2])
CNN_model = nn.models.Model_CNN(0.1)
optimizer = nn.optimizer.MomentumOriginal(init_lr=0.02, model=CNN_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=CNN_model, max_classes=train_labs.max()+1)
val_loss_fn = nn.op.MultiCrossEntropyLoss(max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(CNN_model, optimizer, nn.metric.accuracy, loss_fn, val_loss_fn, scheduler=scheduler)

if isinstance(CNN_model, nn.models.Model_CNN):
    if train_imgs.ndim == 2:
        train_imgs = train_imgs.reshape(-1, 1, 28, 28)
    if valid_imgs.ndim == 2:
        valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=30, log_iters=100, save_dir=r'D:/NN_DL/PJ1/codes/best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()