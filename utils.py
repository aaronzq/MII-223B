import pandas as pd
import numpy as np
import cv2 
import matplotlib
#matplotlib.use('TkAgg')
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

def normalize_u8(img):
    # Normalize the images in the range of 0 to 255 (converted into uint8)
    # 

    return np.uint8( 255*normalize(img) )    

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
    base_r = 0.3
    kernel = np.ones((6,6),np.uint8)

    img_ds = cv2.resize(img, (tar_dim[0],tar_dim[1]), interpolation=cv2.INTER_NEAREST)

    img_norm_u8 = normalize_u8(img_ds)
    img_norm_f32 = normalize(img_ds)


    t, img_bin_base = cv2.threshold(img_norm_u8, round(base_r*255), 255, cv2.THRESH_BINARY)
    base_mask = np.uint8(img_bin_base > 125)


    feature_bin = cv2.GaussianBlur(img_norm_u8*base_mask,(13,13),0)
    feature_bin = cv2.adaptiveThreshold(feature_bin, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17,0)
    bin_mask = np.uint8(feature_bin > 125)
    # opening = cv2.morphologyEx(bin_mask, cv2.MORPH_GRADIENT, kernel)
    # erosion = cv2.erode(bin_mask,kernel,iterations = 1)
    
    # feature_edge = cv2.Canny(feature_bin,50,100)
    # feature_edge = np.uint8(feature_edge > 125)
    # feature_bin = np.float32(feature_bin)

    # img_out = np.dstack((img_norm_u8,feature_bin,feature_edge))

    # plt.axis('off')
    # plt.figure(1)
    # plt.imshow(img_out[:,:,0])
    # plt.figure(2)
    # plt.imshow(img_out[:,:,1])
    # plt.figure(3)
    # plt.imshow(img_out[:,:,2])
    # plt.figure(4)
    # plt.imshow(img_norm_u8*bin_mask)
    # plt.figure(5)
    # plt.imshow(img_norm_u8*opening)
    # plt.show()  

    # img_out = img_to_array(img_norm_f32 * np.float32(bin_mask))
    img_out = img_to_array(img_norm_f32)
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

    rotationNum = 6 # 30 degree as a step

    labelFile = pd.DataFrame(pd.read_csv(labelPath+'Labels-update_final_rm_Inference.csv'))
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

def read_data_seg(labelPath,imgPath,imgDim):
    # read the mask and corresponding images

    rotationNum = 6
    angleInc = 180 // rotationNum

    labelFile = [f for f in listdir(labelPath) if isfile(join(labelPath, f))]
    imgFile = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]

    labelList = list()
    imgList = list()

    for label_f in labelFile:
        label = cv2.imread(join(labelPath,label_f), cv2.IMREAD_GRAYSCALE)
            
        for ang in range(0,180,angleInc):
            labelRot = rotate_bound(label, ang)
            labelPre = myPreprocessing(labelRot,imgDim)
            labelList.append(labelPre)

    for img_f in imgFile:
        img = cv2.imread(join(imgPath,img_f), cv2.IMREAD_GRAYSCALE)

        for ang in range(0,180,angleInc):
            imgRot = rotate_bound(img, ang)
            imgPre = myPreprocessing(imgRot,imgDim)
            imgList.append(imgPre)

    labelout =  np.array(labelList)
    imgout = np.array(imgList)
    
    (train_data, test_data, train_label, test_label) = train_test_split(imgout,
            labelout, test_size=0.2, random_state=42)

    print(' {} Image and {} Mask in total'.format(len(imgList), len(labelList)))
    print(train_data.shape)
    print(train_label.shape)

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

def read_inference_data_gt(imgPath, imgDim):
    # read the all ground truth images under the imgPath 
    files_gt = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
    imgs_gt = list()
    for f in files_gt:
        im = cv2.imread(join(imgPath, f),cv2.IMREAD_GRAYSCALE)
        im = myPreprocessing(im,imgDim)
        imgs_gt.append(im)
    return np.array(imgs_gt) 

def exists_or_mkdir(path):
    # check if the folder exist, if not, create one

    if not isdir(path):
        mkdir(path)

    return 

def save_data_folders(labelPath,imgPath,noNeedle,yesNeedle):
    #  read the labels from a csv file and read the corresponding images
    #  save them in different folders

    labelFile = pd.DataFrame(pd.read_csv(labelPath+'Labels-update_2_rm_Inference.csv'))
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

    labelPath_seg = '../../NeedleImages/Recategorized/yesNeedleMask'
    imgPath_seg = '../../NeedleImages/Recategorized/yesNeedle'

    infPath = '../../NeedleImages/Recategorized/Inference/'


    imgDim=(256,256,1)

    read_data_seg(labelPath_seg,imgPath_seg,imgDim)

    # train_data,train_label,test_data,test_label = read_data(labelPath,imgPath,imgDim)
    # infer_data = read_inference_data(infPath,imgDim)
    
    # print(train_label[0:10])

    # save_data_folders(labelPath,imgPath,noNeedle,yesNeedle)
    # create_label_file(No_Needle,Yes_Needle,savePath)