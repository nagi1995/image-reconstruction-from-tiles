# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:14:56 2021

@author: Nagesh
"""


#%%

import os
import numpy as np
import cv2
import math
import pandas as pd


#%%

def convert_yolo_to_pascal(textfile, size, i, j, scale_h, scale_w):
    annotations = pd.read_csv(textfile, sep = " ", names = ["class", "xc", "yc", "yolo_w", "yolo_h"])
    print(annotations.shape)
    annotations["w"] = annotations["yolo_w"]*size
    annotations["h"] = annotations["yolo_h"]*size
    annotations["x"] = annotations["xc"]*size - annotations["w"]/2 + j*size
    annotations["y"] = annotations["yc"]*size - annotations["h"]/2 + i*size
    
    annotations["w"] = np.round_(annotations["w"]/scale_w)
    annotations["h"] = np.round_(annotations["h"]/scale_h)
    annotations["x"] = np.round_(annotations["x"]/scale_w)
    annotations["y"] = np.round_(annotations["y"]/scale_h)
    
    annotations = annotations[["class", "x", "y", "w", "h"]]
    # annotations.to_csv(textfile, sep=" ", index = False, header = False, float_format = "%.6f")
    return annotations

#%%

def reconstruction_from_tiles(name, size):
    
    
    original_image = "./original/" + name + ".jpg"
    tiled_images = "./tiled/"
    
    im = cv2.imread(original_image)
    h, w, _ = im.shape
    
    
    h_new = math.ceil(h/size) * size
    w_new = math.ceil(w/size) * size
    scale_h = h_new/h
    scale_w = w_new/w
    
    annotations = []
    col_list = []
    for i in range(math.ceil(h/size)):
        
        row_list = []
        for j in range(math.ceil(w/size)):
            tiled_image_name = tiled_images + name + "_" + str(i) + "_" + str(j) + ".jpg"
            im_tiled = cv2.imread(tiled_image_name)
            annot_path = tiled_image_name.replace(".jpg", ".txt")
            annotations.append(convert_yolo_to_pascal(annot_path, size, i, j, scale_h, scale_w))
            row_list.append(im_tiled)
        # Reference: https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
        row_im = cv2.hconcat(row_list)
        col_list.append(row_im)
    
    scaled_image = cv2.vconcat(col_list)
    reconstructed_image = cv2.resize(scaled_image, (w, h), interpolation = cv2.INTER_AREA)
    
    cv2.imwrite("reconstructed_" + name + ".jpg", reconstructed_image)
    df = pd.concat(annotations, axis = 0)
    df.to_csv("reconstructed_" + name + ".txt", sep=" ", index = False, header = False) 
       
    
    for index, row in df.iterrows():
        cv2.rectangle(reconstructed_image, (int(row["x"]), int(row["y"])), (int(row["x"]+row["w"]) , int(row["y"]+row["h"])), (255, 0, 0), 2)
    cv2.imshow("img", reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#%%

if __name__ == "__main__":
    name = "0000006_01111_d_0000003"
    size = 512
    reconstruction_from_tiles(name, size)

