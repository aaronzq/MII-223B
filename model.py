from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras import backend as K
from keras import Model
from keras.applications.resnet50 import ResNet50

def createModel(row,col,depth,classes):
    model = Sequential()
    inputShape = (row, col, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    # print(K.image_data_format())

    model.add(Conv2D(64,(5,5),padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16,(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model


def createModel_AlexNet(row,col,depth,classes):
    model = Sequential()
    inputShape = (row, col, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
    # print(K.image_data_format())

    model.add(Conv2D(96,(11,11),padding='valid', strides = (4,4) ,input_shape = inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (5,5), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(384, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model


def createModel_ResNet(row,col,depth,classes):
	model = Sequential()
	inputShape = (row, col, depth)

	# if we are using "channels first", update the input shape
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
	 # print(K.image_data_format())
	base_model = ResNet50(weights=None, include_top=False, input_shape=inputShape)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.7)(x)
	x = Dense(classes)(x)
	predictions = Activation('softmax')(x)
	
	return Model(inputs=base_model.input,outputs=predictions)



if __name__ == "__main__":
    # model = createModel(256,256,1,2)
    model = createModel_AlexNet(227,227,1,2)
    print(model.summary())