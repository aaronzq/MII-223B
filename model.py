from keras.models import Sequential
from keras.layers import Input, Add, ZeroPadding2D, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as K
from keras import Model
from keras.initializers import glorot_uniform
from keras.applications.resnet50 import ResNet50

def createModel(row,col,depth,classes):
    model = Sequential()
    inputShape = (row, col, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    # print(K.image_data_format())

    model.add(Conv2D(32,(5,5),padding='same', input_shape=inputShape))
    model.add(Activation('relu'))
    # model.add(Conv2D(64,(5,5),padding='valid',strides=(2,2)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(Activation('relu'))
    # model.add(Conv2D(128,(5,5),padding='valid',strides=(2,2)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(Activation('relu'))
    # model.add(Conv2D(128,(5,5),padding='valid',strides=(2,2)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=-1))
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


def identity_block(X, f, filters_num, stage, block):
    """

    X: input tensor
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters_num: list of integers, defining the number of filters in the CONV layersof the main path
    stage: integer, used to mane the layers, depending on the their position in the network
    block: string/character, used to name the layers, depending on their position in the network

    Returns: output of the identity block, tensor 

    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1,F2 = filters_num

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(f,f), strides=(1,1), padding='same', 
        name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
        name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2b')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ## Res50 below
    # X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', 
    #     name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base+'2a')(X)
    # X = Activation('relu')(X)

    # X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
    #     name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base+'2b')(X)
    # X = Activation('relu')(X)

    # X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', 
    #     name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base+'2c')(X)

    # X = Add()([X, X_shortcut])
    # X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters_num, stage, block, s=2):
    """

    X: input tensor
    f: integer, specifying the shape of the middle CONV's window for the main path
    filters_num: list of integers, defining the number of filters in the CONV layersof the main path
    stage: integer, used to mane the layers, depending on the their position in the network
    block: string/character, used to name the layers, depending on their position in the network
    s: Integer, specifying the stride to be used

    Returns: output of the identity block, tensor 

    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1,F2 = filters_num

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(f,f), strides=(s,s), padding='valid', 
        name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
        name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2b')(X)


    X_shortcut = Conv2D(filters=F2, kernel_size=(f,f), strides=(s,s), padding='valid', 
        name = conv_name_base+'1',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base+'1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ## Resnet 50 below
    # X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', 
    #     name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base+'2a')(X)
    # X = Activation('relu')(X)

    # X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
    #     name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base+'2b')(X)
    # X = Activation('relu')(X)

    # X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', 
    #     name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=-1, name=bn_name_base+'2c')(X)


    # X_shortcut = Conv2D(filter=F3, kernel_size=(1,1), strides=(s,s), padding='valid', 
    #     name = conv_name_base+'1',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    # X_shortcut = BatchNormalization(axis=-1, name=bn_name_base+'1')(X_shortcut)

    # X = Add()([X, X_shortcut])
    # X = Activation('relu')(X)

    return X

def createModel_ResNet18(row,col,depth,classes):
    inputShape = (row, col, depth)

	# if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    X_input = Input(inputShape)

    #Stage 1
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=-1, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)

    #Stage 2
    X = convolutional_block(X, f=3, filters_num=[64,64], stage=2, block='a', s=1)
    X = identity_block(X, f=3, filters_num=[64,64], stage=2, block='b')

    #Stage 3
    X = convolutional_block(X, f=3, filters_num=[128,128], stage=3, block='a', s=2)
    X = identity_block(X, f=3, filters_num=[128,128], stage=3, block='b')

    #Stage 4
    X = convolutional_block(X, f=3, filters_num=[256,256], stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters_num=[256,256], stage=4, block='b')

    #Stage 5
    X = convolutional_block(X, f=3, filters_num=[512,512], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters_num=[512,512], stage=5, block='b')

    X = AveragePooling2D((2,2), name='avg_pool')(X)


    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(classes,name='fc'+str(classes),kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet18')

    return model




if __name__ == "__main__":
    model = createModel(64,64,36,2)
    # model = createModel_AlexNet(227,227,1,2)
    # model = createModel_ResNet18(229,229,1,2)
    print(model.summary())