# AI_DocumentLayoutAnalysis
AI_DocumentLayoutAnalysis
Team Name: UIT Departure AI DOCUMENT LAYOUT ANALYSIS
Cinnamon AI Marathon
CONTENT
	About us
	Overview
	Dataset
	Approach to problem solving
I. About us
Team Name: UIT Departure
II. Overview
Designing a system/algorithm to analyze the layout of Vietnamese document image 
(respect to magazine).
Ideas:
-	Sliding Window based on clustering algorithm
-	Sliding Window based on Selective Search window
-	FCN for semantic segmentation
III. Dataset
The contest has provided 200 images and Our team has collected data on the internet for 405 images
Training set: 425 images
Public test set: 120 images
Private test set: 60 images
Training/Evaluation/Test 70:20:10
IV. Approach to problem solving
I. Sliding window
1.Generate data 
Sliding window  with weight x height 140 x 140 Label: (2000 images/label)
•	0: Image  
•	1: Text 
•	2 Heading  
•	3: Background                                       
Training/Evaluation 80:20                                                                       Sliding windown
2 .Result Model using approach sliding window
Optimizer  MomentumSGD
(lr=0.0001, momentum=0.9)
3 hour/epoch
All total time: 4 day
Model : Alexnet , Lenet5
 
	Accuracy	Loss
3. Predict by Sliding Window 140x140 with stride 50 pixel
One problem is that when using this method, we do not know what the suitable sizes of windows are. In this sample, the chosen size is 140 x 140 with 50-pixel pre step.
Predicting for a image is timeconsuming
Another problem is that when using Sliding Window technique in the case of which windows position on 2 objects, predicting would make some unexpected mistakes.
4. How to fix problem
The solution for this case is making a binary image form the original one. The process is completed after considering some attributes such as characteristics of colors, the viewpoint of images. As the results, the predicting for some particles is not necessary, the accuracy increases and it takes less needed time.
However, binarizing process is not always accurate, the result are not good as expected.
5. Selective search for object recognition
Another method, which is selective search for object recognition, is implemented to find out main object-containing windows, which will be material for predicting process. After some test, this method does not give much more better results, because the inaccuracies of two  new processed model make the rate of correctness lower.
 
II. SEMANTIC SEGMENTATION USING DEEP LEARNING
In this case, we choose Fully Convolutional Network-Based Semantic Segmentation method, which improves the negative points of the above method. The results are less needed time and increased accuracy
 
FULLY CONVOLUTIONAL NETWORKS FOR SEMANTIC SEGMENTATION 
1.	Data
From file XML, my team create data (input arbitrary-sized images)
Label: 0 Background
1	TextRegion
2	ImageRegion
3	TableRegion
4	ChartRegion Training:
Learning rate: 0.00001
Loss: 0.04
All time training: 10day
Using gpu nvidia tesla k80
	Origin Images	View Label	Label for AI
FULLY CONVOLUTIONAL NETWORKS ARCHITECTURE
A general semantic segmentation architecture can be broadly thought of as an encoder network followed by a decoder network:
The encoder is usually is a pre-trained classification network like VGG/ResNet followed by a decoder network. A pre-trained VGG16 is used as an encoder
The task of the decoder is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification
The fully connected layers of VGG16 is converted to fully convolutional layers, using 1x1 convolution. 
This process produces a class presence heat map in low resolution.
 
THANK YOU FOR YOUR ATTENTION!
