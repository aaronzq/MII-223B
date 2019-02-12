import pandas as pd
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from skimage.exposure import histogram
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from os import listdir, mkdir
from os.path import isfile, join, isdir


def load_labelNames():
    
    return ['no','yes']

def normalize(img):
    # Normalize the images in the range of 0 to 1 (converted into float64)
    # 512 x 512

    return (img-np.min(img))/(np.max(img)-np.min(img))
    # return img / np.max(img)

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

    
def myPreprocessing(img,tar_dim):
    ## several pre-processing steps

    img_ds = cv2.resize(img, (tar_dim[0],tar_dim[1]))

    feature_bin = cv2.GaussianBlur(img_ds,(13,13),0)
    feature_bin = cv2.adaptiveThreshold(feature_bin, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17,0)
    feature_edge = cv2.Canny(feature_bin,50,100)
    feature_edge = np.float32(feature_edge > 125)
    feature_bin = np.float32(feature_bin)

    img_pre = normalize(img_ds)
    img_out = np.dstack((img_pre,feature_bin,feature_edge))

    plt.axis('off')
    plt.figure(1)
    plt.imshow(img_out[:,:,0])
    plt.figure(2)
    plt.imshow(img_out[:,:,1])
    plt.figure(3)
    plt.imshow(img_out[:,:,2])
    plt.show()  
    img_out = img_to_array(img_out)
    return img_out

def load_image_data(img,label,img_list,label_list,yes_cnt,no_cnt,rotationNum,imgDim):
    ## augment the data with rotation and expand the labels correspondingly

    angleInc = 180 // rotationNum
    for ang in range(0,180,angleInc):
        imgRot = rotate_bound(img, ang)
        imgPre = myPreprocessing(imgRot,imgDim)
        img_list.append(imgPre)
        if label == 'no':
            label_list.append(0)
            no_cnt+=1
        else:
            label_list.append(1)
            yes_cnt+=1
    return yes_cnt,no_cnt


def read_data(labelPath,imgPath,imgDim):
    #  read the labels from a csv file and read the corresponding images

    rotationNum = 3 # 30 degree as a step

    labelFile = pd.DataFrame(pd.read_csv(labelPath+'Labels-update_2_rm_Inference.csv'))
    n,c = labelFile.shape
    img_list=list()
    label_list=list()
    # imgs = np.empty((n,row,col))
    yes=0
    no=0
    for i in range(n):
        ind,imgName,labelName = labelFile.loc[i]
        img = cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE)
        yes,no = load_image_data(img,labelName,img_list,label_list,yes,no,rotationNum,imgDim)

    ##### Transform data into network-compatible format
    data = np.array(img_list, dtype='float')
    labels = np.array(label_list)
    (train_data, test_data, train_label, test_label) = train_test_split(data,
            labels, test_size=0.2, random_state=42)
    # print(train_label[1:10])
    train_label = to_categorical(train_label, num_classes=2)
    # print(train_label[1:10])
    test_label = to_categorical(test_label, num_classes=2)

    # labels = np.eye(2)[np.vstack(label_list).reshape(n,1)]
    print('Containing Needle: {}  ||| Not Containing Needle: {}'.format(yes,no))
    
    return train_data,train_label,test_data,test_label


def read_inference_data(imgPath,imgDim):
    # read the all images under the imgPath and use a trained model to infer
    files_to_infer = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
    imgs_to_infer = list()
    for f in files_to_infer:
        im = cv2.imread(join(imgPath, f),cv2.IMREAD_GRAYSCALE)
        im = myPreprocessing(im,imgDim)
        imgs_to_infer.append(im)
    return np.array(imgs_to_infer)

def exists_or_mkdir(path):
    # check if the folder exist, if not, create one

    if not isdir(path):
        mkdir(path)

    return 

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
    infPath = '../../NeedleImages/Recategorized/Inference/'

    imgDim=(256,256,3)

    train_data,train_label,test_data,test_label = read_data(labelPath,imgPath,imgDim)
    infer_data = read_inference_data(infPath,imgDim)
    
    print(train_label[0:10])

    # test = train_data[8,:,:,0]
    # hist, hist_centers = histogram(test)

    # test2 = train_data[9,:,:,0]
    # hist2, hist_centers2 = histogram(test2)
    # print('++++++')
    # print(hist)
    # print('++++++')
    # print(hist_centers)
    # testbin = cv2.adaptiveThreshold(test,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,12,3)
    # testedge = cv2.Canny(testbin)


    # plt.axis('off')
    # plt.figure(1)
    # plt.imshow(test)
    # plt.figure(2)
    # plt.plot(hist_centers,hist,lw=2)
    # plt.figure(3)
    # plt.imshow(test2)
    # plt.figure(4)
    # plt.plot(hist_centers2,hist2,lw=2)
    # plt.figure(5)
    # plt.imshow(testbin)
    # plt.figure(6)
    # plt.imshow(testedge)
    # plt.show()




    # pass
    # imgName = '18954.jpg'
    # im_test = cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(im_test, (256,256))
    # # img = normalize(img)
    # imgh = cv2.GaussianBlur(img,(5,5),1)
    # imgl = cv2.GaussianBlur(img,(15,15),5)
    # img = np.abs(imgh-imgl)
    # imgbin = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17,0)
    # imgedge = cv2.Canny(imgbin,25,100)

    # plt.axis('off')
    # plt.figure(1)
    # plt.imshow(img)
    # plt.figure(2)
    # plt.imshow(imgbin)
    # plt.figure(3)
    # plt.imshow(imgedge)
    # plt.show()
    # save_data_folders(labelPath,imgPath,noNeedle,yesNeedle)
    # create_label_file(No_Needle,Yes_Needle,savePath)