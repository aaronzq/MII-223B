import pandas as pd
import numpy as np
import cv2 
import matplotlib.pyplot as plt 

row=512
col=512

def load_labelNames():
    
    return ['no','yes']

def normalize(img):
    # Normalize the images in the range of 0 to 1 (converted into float64)
    # 512 x 512

    return (img-np.min(img))/(np.max(img)-np.min(img)) 

def read_data(labelPath,imgPath):
    #  read the labels from a csv file and read the corresponding images

    labelFile = pd.DataFrame(pd.read_csv(labelPath+'Labels.csv'))
    n,c = labelFile.shape
    img_list=list()
    label_list=list()
    imgs = np.empty((n,row,col))
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
        imgs[i,:,:] = normalize(cv2.imread(imgPath+imgName,cv2.IMREAD_GRAYSCALE))
    labels = np.eye(2)[np.vstack(label_list).reshape(n,1)]
    print('Containing Needle: {}  ||| Not Containing Needle: {}'.format(yes,no))
    
    return imgs,labels

if __name__ == "__main__":
    labelPath = './NeedleImages/'
    imgPath = './NeedleImages/'

    imgs,labels = read_data(labelPath,imgPath)
    pass

    # plt.axis('off')
    # plt.imshow(imgs[150])
    # plt.show()


