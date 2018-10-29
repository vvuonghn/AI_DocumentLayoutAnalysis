import numpy as np
from PIL import Image
path= 'E:\\MyData\\Project\\Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation-master\\AI_3\\Label\\'

img1 = Image.open(path+'00000089.png')
img2 = Image.open(path+'00000086.png')
list1 = img1.getdata()
list2 = img2.getdata()
duplicate = 0
for i in range(len(list1)):
    if list1[i] == list2[i]:
        duplicate += 1
print(format(duplicate/len(list1)*100, '0.2f')+' %')
