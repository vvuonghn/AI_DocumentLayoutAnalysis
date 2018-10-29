import os
import cv2
import numpy as np
Inpath='E:\\MyData\\Project\\Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation-master\\AI_3\\Label\\'
Outpath='E:\\MyData\\Project\\Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation-master\\AI_3\\NewLabel\\'
filenames = os.listdir(Inpath)

for f in filenames:
	infile=Inpath+f
	outfile=Outpath+f
	label=cv2.imread(infile)
	label[label==4] = 2
	label[label==3] = 1
	cv2.imwrite(outfile,label)
	print(infile)