from utils import read_data,read_inference_data,exists_or_mkdir
from keras_preprocessing.image import ImageDataGenerator
from model import createModel, createModel_AlexNet, createModel_ResNet, createModel_ResNet18
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="train or infer")
args = vars(ap.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

labelPath = '../../NeedleImages/Recategorized/'
imgPath = '../../NeedleImages/Recategorized/'
infPath = '../../NeedleImages/Recategorized/Inference/'
savePath = './save/'
exists_or_mkdir(savePath)
checkpointPath = './save/checkpoint/'
exists_or_mkdir(checkpointPath)

imgDim = (229,229,1)
classNum = 2
batchSize = 32
epochsnum = 200
INIT_LR = 6e-4

def train():
	train_data,train_label,test_data,test_label = read_data(labelPath,imgPath,imgDim)
	aug = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0, zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True)

    # model = createModel(*imgDim,classNum)
	# model = createModel_AlexNet(*imgDim,classNum)
	# model = createModel_ResNet(*imgDim,classNum)
	model = createModel_ResNet18(*imgDim,classNum)
	
	opt = Adam(lr=INIT_LR,decay=INIT_LR / epochsnum)
	# opt = Adam(lr=INIT_LR)

	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])

	# checkpoint
	ckptName = 'checkpoint.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.model'
	checkpoint = ModelCheckpoint(checkpointPath+ckptName, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=20)
	callbacks_list = [checkpoint]

	# fit the data
	H = model.fit_generator(aug.flow(train_data,train_label,batch_size=batchSize,shuffle=True),
	validation_data=(test_data,test_label), steps_per_epoch=len(train_data)//batchSize,
	epochs=epochsnum, verbose=1, callbacks=callbacks_list)


	## Save the model and plot
	model.save(savePath+'needle.model')
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

def infer():
	model = load_model(savePath+'needle.model')
	imgInfer = read_inference_data(infPath,imgDim)
	result = model.predict(imgInfer)
	print(result)
	for nneed, need in result:
		if need > nneed:
			print('Needle!')
		elif need < nneed:
			print('No needle!')
		else:
			print('Hard to tell')

if __name__ == "__main__":
	if args['model'] == 'train':
		train()
	elif args['model'] == 'infer':
		infer()
	else:
		print('Input correct parameters: train or infer')