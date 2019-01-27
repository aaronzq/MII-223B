from utils import read_data
from keras_preprocessing.image import ImageDataGenerator
from model import createModel
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model

labelPath = '../../NeedleImages/Recategorized/'
imgPath = '../../NeedleImages/Recategorized/'
savePath = './save/'

imgDim = (256,256,1)
classNum = 2
batchSize = 32
epochsnum = 100
INIT_LR = 3e-4

train_data,train_label,test_data,test_label = read_data(labelPath,imgPath,imgDim)

## construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output accuracy/loss plot")
# args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0, zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True)

model = createModel(*imgDim,classNum)
# opt = Adam(lr=INIT_LR,decay=INIT_LR / epochsnum)
opt = Adam(lr=INIT_LR)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])

H = model.fit_generator(aug.flow(train_data,train_label,batch_size=batchSize,shuffle=True),
    validation_data=(test_data,test_label), steps_per_epoch=len(train_data)//batchSize,
    epochs=epochsnum, verbose=1)

model.save(savePath+'needle.model')
# model = load_model(savePath+'needle.model')
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochsnum
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Needle/not Needle")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(savePath+'plot.png')
