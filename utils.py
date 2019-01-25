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

    return (img-np.min(img))/(np.max(img)-np.min(img))
    # return img / np.max(img)

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
        img = cv2.resize(img, (256,256))
        # img = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
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

def save_data_folders(labelPath,imgPath,noNeedle,yesNeedle):
    #  read the labels from a csv file and read the corresponding images
    #  save them in different folders

    labelFile = pd.DataFrame(pd.read_csv(labelPath+'Labels.csv'))
    n,c = labelFile.shape
    for i in range(n):
        ind,imgName,labelName = labelFile.loc[i]
        img = cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE)
        if labelName == 'no':            
            cv2.imwrite(noNeedle+imgName,img)
        else:
            cv2.imwrite(yesNeedle+imgName,img)

def create_label_file(No_Needle,Yes_Needle,savePath):
    from os import listdir
    from os.path import isfile, join
        
    col1 = 'ind' 
    col2 = 'External ID'
    col3 = 'Label'

    nd_files = [f for f in listdir(No_Needle) if isfile(join(No_Needle, f))]
    nd_labels = ['no' for i in range(0,len(nd_files))]
    yd_files = [f for f in listdir(Yes_Needle) if isfile(join(Yes_Needle, f))]
    yd_labels = ['yes' for i in range(0,len(yd_files))]

    EID = nd_files + yd_files
    L = nd_labels + yd_labels
    
    df = {}
    df[col2] = EID
    df[col3] = L

    csvSAVE = pd.DataFrame(df, columns = [col2,col3])
    csvSAVE.to_csv(join(savePath,'Labels.csv'))

    
if __name__ == "__main__":

    labelPath = '../../NeedleImages/Recategorized/'
    imgPath = '../../NeedleImages/Recategorized/'
    savePath = '../../NeedleImages/Recategorized/'

    train_data,train_label,test_data,test_label = read_data(labelPath,imgPath)
    
    print(train_label[1:10])


    # pass
    # imgName = '18954.jpg'
    # im_test = cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(im_test, (256,256))
    # # img = normalize(img)
    # img = cv2.GaussianBlur(img,(5,5),0)
    # imgbin = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,13,0)
    # imgedge = cv2.Canny(imgbin,25,100)

    # plt.axis('off')
    # plt.figure(1)
    # plt.imshow(img)
    # plt.figure(2)
    # plt.imshow(imgbin)
    # plt.figure(3)
    # plt.imshow(imgedge)
    # plt.show()
    # ave_data_folders(labelPath,imgPath,noNeedle,yesNeedle)
    # create_label_file(No_Needle,Yes_Needle,savePath)