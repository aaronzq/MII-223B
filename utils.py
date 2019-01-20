import pandas as pd
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

def load_labelNames():
    
    return ['no','yes']

def normalize(img):
    # Normalize the images in the range of 0 to 1 (converted into float64)
    # 512 x 512

    # return (img-np.min(img))/(np.max(img)-np.min(img))
    return img / np.max(img)

def myPreprocessing(img):

    return []


def read_data(labelPath,imgPath):
    #  read the labels from a csv file and read the corresponding images

    labelFile = pd.DataFrame(pd.read_csv(labelPath+'Labels.csv'))
    n,c = labelFile.shape
    img_list=list()
    label_list=list()
    # imgs = np.empty((n,row,col))
    yes=0
    no=0
    for i in range(n):
        ind,imgName,labelName = labelFile.loc[i]
        if labelName == 'no':
            label_list.append(0)
            no+=1
        else:
            label_list.append(1)
            yes+=1
        img = cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128,128))
        img = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
        img = normalize(img)
        img = img_to_array(img)
        img_list.append(img)
        # imgs[i,:,:] = normalize(cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE))

    data = np.array(img_list, dtype='float')
    labels = np.array(label_list)
    (train_data, test_data, train_label, test_label) = train_test_split(data,
            labels, test_size=0.1, random_state=42)
    # print(train_label[1:10])
    train_label = to_categorical(train_label, num_classes=2)
    # print(train_label[1:10])
    test_label = to_categorical(test_label, num_classes=2)

    # labels = np.eye(2)[np.vstack(label_list).reshape(n,1)]
    print('Containing Needle: {}  ||| Not Containing Needle: {}'.format(yes,no))
    
    return train_data,train_label,test_data,test_label

if __name__ == "__main__":
    labelPath = '../../NeedleImages/'
    imgPath = '../../NeedleImages/'

    # train_data,train_label,test_data,test_label = read_data(labelPath,imgPath)
    # pass
    # print(train_label[1:10])
    # print(train_label[20])


    pass
    imgName = '14717.jpg'
    im_test = cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(im_test, (256,256))
    # img = normalize(img)
    # img = cv2.GaussianBlur(img,(5,5),0)
    cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
    # img = cv2.Canny(img,25,100)

    plt.axis('off')
    plt.imshow(img)
    plt.show()