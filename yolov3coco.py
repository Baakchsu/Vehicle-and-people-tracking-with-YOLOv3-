# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2017 Taehoon Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Created on Mon Sep 10 19:40:49 2018

@author: Baakchsu
"""


import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time


inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
#model = nets.YOLOv2(inputs, nets.Darknet19)

#frame=cv2.imread("D://pyworks//yolo//truck.jpg",1)

classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
list_of_classes=[0,1,2,3,5,7]
with tf.Session() as sess:
    sess.run(model.pretrained())
#"D://pyworks//yolo//videoplayback.mp4"    
    cap = cv2.VideoCapture("G://ani_pers//videoplayback.mp4")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        img=cv2.resize(frame,(416,416))
        imge=np.array(img).reshape(-1,416,416,3)
        start_time=time.time()
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
    
        print("--- %s seconds ---" % (time.time() - start_time)) 
        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)

        cv2.resizeWindow('image', 700,700)
        #print("--- %s seconds ---" % (time.time() - start_time)) 
        boxes1=np.array(boxes)
        for j in list_of_classes:
            count =0
            if str(j) in classes:
                lab=classes[str(j)]
            if len(boxes1) !=0:
                
                
                for i in range(len(boxes1[j])):
                    box=boxes1[j][i] 
                    
                    if boxes1[j][i][4]>=.40:
                        
                            
                        count += 1    
 
                        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
                        cv2.putText(img, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), lineType=cv2.LINE_AA)
            print(lab,": ",count)
    
              
        cv2.imshow("image",img)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          




cap.release()
cv2.destroyAllWindows()    
