#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
# import streamlit as st
from PIL import Image,ImageFilter
import matplotlib.image as mpimg
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

#====================== READ A INPUT IMAGE =========================


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image') 
plt.axis ('off')
plt.show()


#============================ PREPROCESS =================================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
         
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()



#============================ IMAGE SPLITTING =============================


import os 

# === test and train ===

from sklearn.model_selection import train_test_split

data_fire = os.listdir('Data/Fire/')


data_nofire = os.listdir('Data/NoFire/')



dot1= []
labels1 = []
for img1 in data_fire:
        # print(img)
        img_1 = mpimg.imread('Data/Fire/' + "/" + img1)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

        
for img1 in data_nofire:
    try:
        img_2 = mpimg.imread('Data/NoFire/'+ "/" + img1)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print("---------------------------------------------------")
print("Image Splitting")
print("---------------------------------------------------")
print()

print("Total no of input data   :",len(dot1))
print("Total no of train data   :",len(x_train))
print("Total no of test data    :",len(x_test))    



# ======================= YOLO 




# ------ YOLO v7


import time
import cv2
import numpy as np
import onnxruntime

from yolov7.utils import xywh2xyxy, nms, draw_detections


class YOLOv7:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, official_nms=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.official_nms = official_nms

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

        self.has_postprocess = 'score' in self.output_names or self.official_nms


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):

        #Pinto's postprocessing is different from the official nms version
        if self.official_nms:
            scores = outputs[0][:,-1]
            predictions = outputs[0][:, [0,5,1,2,3,4]]
        else:
            scores = np.squeeze(outputs[0], axis=1)
            predictions = outputs[1]
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1].astype(int)
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        if not self.official_nms:
            boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]



# ------------------- DETECTION


import xml.etree.ElementTree as ET
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 

try:

    inpimg = mpimg.imread(filename)
    
    aa= filename.split('/')
    
    aa3 = aa[len(aa)-1]
    
    
    ff = 'Data/Annotation/'+str(aa3[0:len(aa3)-4])+'.xml'
    
    
    # ff = filename[0:len(filename)-4]+str('.xml')
    xml_data = open(ff, 'r').read()  # Read file
    root = ET.XML(xml_data)  # Parse XML
    
    data = []
    cols = []
    vval = []
    cols1 = []
    vval2 = []
    
    for i, child in enumerate(root):
        data.append([subchild.text for subchild in child])
        cols.append(child.tag)
        vval2.append([subchildds.text for subchildds in child])
        vval.append([subchildd.tag for subchildd in child])
    
    
    
    import pandas as pd
    
    df = pd.DataFrame(data).T  
    df.columns = cols  
    
    df1 = pd.DataFrame(vval).T  
    df1.columns = cols 
    
    
    import xmltodict
    import pandas as pd
    
    xml_data = open(ff, 'r').read() 
    xmlDict = xmltodict.parse(xml_data)  
    
    colsss = xmlDict['annotation']
    
    try:    
        AZ = inpimg 
        for iii in range(0,len(colsss['object'])):
            
            Dims = colsss['object'][iii]['bndbox']
            
            D1 = int(Dims['xmax'])
            D2 = int(Dims['xmin'])
            D3 = int(Dims['ymax'])
            D4 = int(Dims['ymin'])
            
            import cv2
            AZ = cv2.rectangle(AZ, (D1, D4), (D2, D3), (255,0,0), 3)
    except:
        
        Dims = colsss['object']['bndbox']
        
        D1 = int(Dims['xmax'])
        D2 = int(Dims['xmin'])
        D3 = int(Dims['ymax'])
        D4 = int(Dims['ymin'])
        
        import cv2
        AZ = cv2.rectangle(inpimg, (D1, D4), (D2, D3), (255,0,0), 3)
        
    plt.imshow(AZ)
    plt.title('DETECTED IMAGE')
    plt.show()
    
    try:
        DD = df['object']
        print("------------------------------")
        print(" IDENTIFIED  = ",DD[0])
        print("------------------------------")
    
        # print(DD[0])
    except:
        print(' ')
        aa = df['object'][:1]
        print(aa)


except:
    
    print("Non Fire")




# import yolov5

# # load pretrained model
# model = yolov5.load('yolov5s.pt')

# or load custom model
# model = yolov5.load('train/best.pt')
  
# # set model parameters
# model.conf = 0.25  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per image

# # set image
# img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# # perform inference
# results = model(img)

# # inference with larger input size
# results = model(img, size=1280)

# # inference with test time augmentation
# results = model(img, augment=True)

# # parse results
# predictions = results.pred[0]
# boxes = predictions[:, :4] # x1, y1, x2, y2
# scores = predictions[:, 4]
# categories = predictions[:, 5]

# # show detection bounding boxes on image
# results.show()

# # save results into "results/" folder
# results.save(save_dir='results/')

# ========================= CLASSIFICATION ================================

from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]

print("-------------------------------------------------------------")
print('Convolutional Neural Network') 
print("-------------------------------------------------------------")
print()
print()


# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=10,verbose=1)
accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)

print()
print()
print("-------------------------------------------------------------")
print("Performance Analysis")
print("-------------------------------------------------------------")
print()

loss=history.history['loss']
loss=max(loss)
accuracy=100-loss

print()
print("1.Accuracy    :",accuracy,'%')
print()
print("2.Error Rate  :",loss,'%')
print()


# ====================== PREDICTION ===================



Total_length = len(data_fire) + len(data_nofire) 

temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 0:
    print('------------------------')
    print(' IDENTIFIED = FIRE ')
    print('------------------------')
else:
    print('------------------------')
    print(' IDENTIFIED= NO FIRE ')
    print('------------------------')





