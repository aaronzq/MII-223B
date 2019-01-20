from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as K

def createModel(row,col,depth,classes):
    model = Sequential()
    inputShape = (row, col, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    print(K.image_data_format())

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",
        input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model


    # model = Sequential()
    # model.add(Conv2D(64,(5,5),padding='same', input_shape=(row,col,depth)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32,(5,5)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    # # model.add(Conv2D(32,(5,5),padding='same'))
    # # model.add(Activation('relu'))
    # # model.add(Conv2D(32,(5,5)))
    # # model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2,2)))
    # # model.add(Dropout(0.2))

    # # model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    # # model.add(Conv2D(64,(3,3),activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2,2)))
    # # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(classes))
    # model.add(Activation('softmax'))

    # return model

if __name__ == "__main__":
    model = createModel(128,128,1,2)
    print(model.summary())