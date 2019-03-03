from utils import read_data,read_inference_data,exists_or_mkdir, read_data_seg
from keras_preprocessing.image import ImageDataGenerator
from model import createModel, createModel_AlexNet, createModel_ResNet, createModel_ResNet18, createModel_DensNet, createModel_Unet
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="train or infer")
args = vars(ap.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

labelPath = '../../NeedleImages/Recategorized/'
imgPath = '../../NeedleImages/Recategorized/'
infPath = '../../NeedleImages/Recategorized/Inference/'
labelPath_seg = '../../NeedleImages/Recategorized/yesNeedleMask'
imgPath_seg = '../../NeedleImages/Recategorized/yesNeedle'
savePath = './save/'
exists_or_mkdir(savePath)
checkpointPath = './save/checkpoint/'
exists_or_mkdir(checkpointPath)
tfbdPath = './save/tensorboard/'
exists_or_mkdir(tfbdPath)


def train():
	imgDim = (224,224,1)
	classNum = 2
	batchSize = 32
	epochsnum = 200
	INIT_LR = 6e-4

	train_data,train_label,test_data,test_label = read_data(labelPath,imgPath,imgDim)
	aug = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0, zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True)

	# model = createModel(*imgDim,classNum)  #256
	# model = createModel_AlexNet(*imgDim,classNum) #227
	# model = createModel_ResNet(*imgDim,classNum)  #227
	# model = createModel_ResNet18(*imgDim,classNum)  #229
	model = createModel_DensNet(*imgDim,classNum)  #224
	
	opt = Adam(lr=INIT_LR,decay=INIT_LR / epochsnum)
	# opt = Adam(lr=INIT_LR)

	#model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
	model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=["categorical_accuracy"])


	tfbd = TensorBoard(log_dir=tfbdPath, histogram_freq=0,write_graph=True, write_images=True)

	# checkpoint
	ckptName = 'checkpoint{epoch:02d}-loss{loss:.2f}-val_loss{val_loss:.2f}-acc{categorical_accuracy:.2f}-val_acc{val_categorical_accuracy:.2f}.model'
	checkpoint = ModelCheckpoint(checkpointPath+ckptName, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max', period=4)
	
	callbacks_list = [checkpoint,tfbd]

	# fit the data
	H = model.fit_generator(aug.flow(train_data,train_label,batch_size=batchSize,shuffle=True),
	validation_data=(test_data,test_label), steps_per_epoch=len(train_data)//batchSize,
	epochs=epochsnum, verbose=1, callbacks=callbacks_list)

	#print(H.history)
	## Save the model and plot
	model.save(savePath+'needle.model')
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = epochsnum
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["categorical_accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_categorical_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy on Needle/not Needle")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(savePath+'plot.png')

def infer():
	imgDim = (224,224,1)

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

def train_seg():
	imgDim = (256,256,1)
	batchSize = 32
	epochsnum = 200
	INIT_LR = 6e-4

	train_data,train_label,test_data,test_label = read_data_seg(labelPath_seg,imgPath_seg,imgDim)

	model = createModel_Unet(*imgDim)  #256
	
	opt = Adam(lr=INIT_LR,decay=INIT_LR / epochsnum)
	# opt = Adam(lr=INIT_LR)

	model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
	# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=["categorical_accuracy"])

	#tensorboard
	tfbd = TensorBoard(log_dir=tfbdPath, histogram_freq=0,write_graph=True, write_images=True)

	# checkpoint
	ckptName = 'checkpoint{epoch:02d}-loss{loss:.2f}-val_loss{val_loss:.2f}-acc{acc:.2f}-val_acc{val_acc:.2f}.model'
	checkpoint = ModelCheckpoint(checkpointPath+ckptName, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=4)
	
	callbacks_list = [checkpoint,tfbd]

	# fit the data
	H = model.fit(train_data,train_label, batch_size=batchSize, validation_data=(test_data,test_label), epochs=epochsnum, verbose=1, callbacks=callbacks_list)

	#print(H.history)
	## Save the model and plot
	model.save(savePath+'needle_seg.model')
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


if __name__ == "__main__":
	if args['model'] == 'train':
		# train()
		train_seg()
	elif args['model'] == 'infer':
		infer()
	else:
		print('Input correct parameters: train or infer')