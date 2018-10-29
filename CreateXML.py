from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

def filter(image, kernel_er, kernel_di):
  kernel_1 = np.ones((kernel_er,kernel_er),np.uint8)
  kernel_2 = np.ones((kernel_di,kernel_di),np.uint8)

  image = cv2.erode(image,kernel_1,iterations = 1)
  image = cv2.dilate(image,kernel_2,iterations = 1)
  return image

def getPoint_version1_org(label):
    Listlabel1 = []
    Listlabel2 = []
    Listlabel3 = []
    Listlabel4 = []
    # label = cv2.imread(pathlabel)

    for i in range(1, 5, 1):
        labeli = label.copy()
        labeli[labeli != i] = 0
        labeli[labeli == i] = 255
        labeli=filter(labeli,7,7)
        # if i==1:
        #     labeli=filter(labeli,7,9)
        #     plt.imshow(labeli)
        #     plt.show()
        # labeli = labeli[:, :, 0]
        w, h = labeli.shape
        _, labeli = cv2.threshold(labeli, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(labeli, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for idx in contours:
            if len(idx) <= 300 or cv2.contourArea(idx) <= 400:
                continue
            a = np.asarray(idx)
            a = np.rollaxis(a, 1, 3)
            a = np.reshape(a, (len(a), 2))
            if i == 1:
                Listlabel1.append(a)
            if i == 2:
                Listlabel2.append(a)
            if i == 3:
                Listlabel3.append(a)
            if i == 4:
                Listlabel4.append(a)
    return (Listlabel1, Listlabel2, Listlabel3, Listlabel4)

def getPoint(label):
    Listlabel1 = []
    Listlabel2 = []
    Listlabel3 = []
    Listlabel4 = []
    w,h=label.shape
    img = np.zeros((w,h,3), dtype=np.uint8)
    # label = cv2.imread(pathlabel)
    label1 = label.copy()
    label1[label1 != 1] = 0
    label1[label1 == 1] = 255
    label1=filter(label1,7,9) 

    label2 = label.copy()
    label2[label2 != 2] = 0
    label2[label2 == 2] = 255
    _, label2 = cv2.threshold(label2, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(label2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for idx in range(len(contours)):
        label2 = cv2.fillPoly(label2, pts =[contours[idx]], color=255)

    label3 = label.copy()
    label3[label3 != 3] = 0
    label3[label3 == 3] = 255

    label4 = label.copy()
    label4[label4 != 4] = 0
    label4[label4 == 4] = 255
    label1=label1-label2
    label2=filter(label2,7,5) 
    lable1234=['',label1,label2,label3,label4]
    for i in range(1, 5, 1):
        _, lable1234[i] = cv2.threshold(lable1234[i], 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(lable1234[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for idx in contours:
            if len(idx) <= 200 or cv2.contourArea(idx) <= 300:
                continue
            a = np.asarray(idx)
            a = np.rollaxis(a, 1, 3)
            a = np.reshape(a, (len(a), 2))
            if i == 1:
                Listlabel1.append(a)
            if i == 2:
                Listlabel2.append(a)
            if i == 3:
                Listlabel3.append(a)
            if i == 4:
                Listlabel4.append(a)
    return (Listlabel1, Listlabel2, Listlabel3, Listlabel4)

def createMeta(infoSrcImage):
    meta = '<?xml version="1.0" encoding="UTF-8"?>\n' \
           '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15" ' \
           'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
           'xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15 ' \
           'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd">' \
           '\n\t<Metadata>\n\t' \
           '<Creator>UIT Departure</Creator>\n\t' \
           '<Created>'
    meta += datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    meta += '</Created>\n\t'
    meta += '<LastChange>'
    meta += datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    meta += '</LastChange></Metadata>\n\t'
    meta += '<Page imageFilename=\"'
    meta += infoSrcImage[0]
    meta += '\" imageWidth=\"'
    meta += str(infoSrcImage[2])
    meta += '\" imageHeight=\"'
    meta += str(infoSrcImage[1])
    meta += '\">'

    return meta


def createRegion(contour, lable, id):
    region = '\n\t<' + str(lable) + ' id=\"' + str(id) + '\"'
    if str(lable) == 'TextRegion':
        region += ' type=\"paragraph\"'
    region += '>\n\t<Coords points=\"'
    region += str(contour[0][0]) + ',' + str(contour[0][1])
    for idx in range(1,len(contour)):
        temp = ' '+str(contour[idx][0]) + ',' + str(contour[idx][1])
        region += temp
    region += '\"/>'
    if lable == 'TextRegion':
        region += '\n\t<TextEquiv>\n\t<Unicode></Unicode></TextEquiv>'
    region += '</' + str(lable) + '>'
    return region

def getLabe(index):
    if index == 0:
        return 'TextRegion'
    if index == 1:
        return 'ImageRegion'
    if index == 2:
        return 'TableRegion'
    if index == 3:
        return 'ChartRegion'
    return ''


def writeXML(filename, infoSrcImage, contours):
    XML = open(filename, 'w')
    meta = createMeta(infoSrcImage)
    XML.write(meta)
    index = 0
    for idx in range(len(contours)):
        lable = getLabe(idx)
        for contour in contours[idx]:
            id = 'r' + str(index)
            index += 1
            XML.write(createRegion(contour, lable, id))
    XML.write('</Page></PcGts>')
    XML.close()


def SaveXML(img,filename,path_XML):
    w, h = img.shape
    contours = getPoint(img)
    saveXML=path_XML+filename[:-4]+'.xml'
    infoSrcImage = (filename, w, h)
    writeXML(saveXML, infoSrcImage, contours)

# path = 'C:\\Users\\V A N L O C\\Desktop\\Preprocess\\Label\\00000122.png'
# contours = getPoint(path)
# img = cv2.imread(path)
# w, h, _ = img.shape
# filename = path.split('\\')[-1]
# infoSrcImage = (filename, w, h)
# writeXML(filename.split('.')[0] + '.xml', infoSrcImage, contours)
