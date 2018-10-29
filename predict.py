# Run prediction and genertae pixelwise annotation for every pixels in the image using fully coonvolutional neural net
# Output saved as label images, and label image overlay on the original image
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set number of classes number in NUM_CLASSES
# 4) Set Pred_Dir the folder where you want the output annotated images to be save
# 5) Run script
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import TensorflowUtils
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
import CheckVGG16Model
import cv2
import matplotlib.pyplot as plt
import time
import CreateXML
ListSize=[]
import shutil
print('Run file predict')

def ResizeImage(pathRead,pathSave,pathBinary):
    global width_ORG,height_ORG
    fileImage = os.listdir(pathRead)
    for file in fileImage:
        img=pathRead+file
        imgsave=pathSave+file
        binarysave=pathBinary+file
        IMG=cv2.imread(img)
        width_ORG, height_ORG, _ = IMG.shape
        size=(width_ORG, height_ORG)
        ListSize.append(size)
        imgBinary = np.zeros((width_ORG,height_ORG,3), dtype=np.uint8)
        # print('Images shape befor resize:  ',IMG.shape)
        IMG = cv2.resize(IMG, (0,0), fx=0.33, fy=0.33)
        # print('Images shape after resize:  ',IMG.shape)
        cv2.imwrite(binarysave,imgBinary)
        cv2.imwrite(imgsave,IMG)

logs_dir= "E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\Model_logs\\"# "path to logs directory where trained model and information will be stored"
Image_Read="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\Input_Image\\"
Image_Dir="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\ImageResize\\"# Test image folder
Image_Binary="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\Binary\\"
path_XML='E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\Output_XML\\'
Pred_Dir="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_3\\OutputTest\\" # Library where the output prediction will be written
model_path="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\Model_logs\\"# "Path to pretrained vgg16 model for encoder"
# path_mask="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\AI_DocumentLayoutAnalysis\\mask\\"
NameEnd="" # Add this string to the ending of the file name optional
NUM_CLASSES = 5 # Number of classes
w=0.4# weight of overlay on image
if not os.path.exists(Image_Dir): os.makedirs(Image_Dir)


print('Resize')
ResizeImage(Image_Read,Image_Dir,Image_Binary)
print(ListSize)
#-------------------------------------------------------------------------------------------------------------------------
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it

################################################################################################################################################################################
def main(argv=None):
      # .........................Placeholders for input image and labels........................................................................
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB

    # -------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    Net.build(image, NUM_CLASSES, keep_prob)  # Build net and load intial weights (weights before training)
    # -------------------------Data reader for validation/testing images-----------------------------------------------------------------------------------------------------------------------------
    ValidReader = Data_Reader.Data_Reader(Image_Dir,  BatchSize=1)
    # print(ValidReader)

    # exit()
    #-------------------------Load Trained model if you dont have trained model see: Train.py-----------------------------------------------------------------------------------------------------------------------------

    sess = tf.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        sys.exit()

#--------------------Create output directories for predicted label, one folder for each granulairy of label prediciton---------------------------------------------------------------------------------------------------------------------------------------------

    if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
    if not os.path.exists(Pred_Dir+"/OverLay"): os.makedirs(Pred_Dir+"/OverLay")
    if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")

    
    print("Running Predictions:")
    print("Saving output to:" + path_XML)
 #----------------------Go over all images and predict semantic segmentation in various of classes-------------------------------------------------------------
    fim = 0
    print("Start Predicting " + str(ValidReader.NumFiles) + " images")

    startTime=time.time()
    while (ValidReader.itr < ValidReader.NumFiles):

        # ..................................Load image.......................................................................................
        FileName=ValidReader.OrderedFiles[ValidReader.itr] #Get input image name
        Images = ValidReader.ReadNextBatchClean()  # load testing image

        # Predict annotation using net
        LabelPred = sess.run(Net.Pred, feed_dict={image: Images, keep_prob: 1.0})
        #------------------------Save predicted labels overlay on images---------------------------------------------------------------------------------------------
        endTimePredict=time.time()
        print('\n\nTime predict image',FileName ,' : ',  endTimePredict - startTime)
        ImageResult=Images[0].copy()
        LabelResult=LabelPred[0].copy()

        # print('Label shape :  ',LabelResult.shape)
        # print('Images shape:  ',ImageResult.shape)
        LabelResult=LabelResult.astype(np.uint8)

        # print('width_ORG,height_ORG', width_ORG,height_ORG)
        # print('Images shape after resize:  ',ImageResult.shape)

        imgORG=cv2.imread(Image_Read+FileName)
        height_ORG,width_ORG,_=imgORG.shape
        print('imgORG shape ', imgORG.shape)
        print('ListSize : ',ListSize[fim][1],'       ',ListSize[fim][0])

        ImageResult = cv2.resize(ImageResult, (width_ORG, height_ORG)) 
        # LabelResult = cv2.resize(LabelResult, (ListSize[fim][1], ListSize[fim][0]))
        LabelResult = cv2.resize(LabelResult, (width_ORG, height_ORG))
        
        # file_mask=path_mask+FileName 
        # cv2.imwrite(file_mask,LabelResult)
        # ImageResult = cv2.resize(ImageResult, (height_ORG, width_ORG)) 
        # LabelResult = cv2.resize(LabelResult, (height_ORG, width_ORG))
        # print('min LabelResult',LabelResult.min())
        # print('max LabelResult',LabelResult.max())
        
        
        

        CreateXML.SaveXML(LabelResult,FileName,path_XML)
        print('Time CreateXML image',FileName ,' : ',time.time() - endTimePredict)
        startTime=time.time()
        fim += 1
        print('Processing : ',str(fim * 100.0 / ValidReader.NumFiles) + "%")
'''
        misc.imsave(Pred_Dir + "/OverLay/"+ FileName+NameEnd  , Overlay.OverLayLabelOnImage(Images[0],LabelPred[0], w)) #Overlay label on image
        misc.imsave(Pred_Dir + "/Label/" + FileName[:-4] + ".png" + NameEnd, LabelPred[0].astype(np.uint8))
        # misc.imsave(Pred_Dir + "/Label/" + FileName[:-4] + ".png" + NameEnd, LabelResult)
''' 
        ##################################################################################################################################################
main()#Run script
shutil.rmtree(Image_Dir, ignore_errors=True)
print('Now time: ',time.strftime("%H:%M:%S"))
