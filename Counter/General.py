# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:24:26 2023

@author: adlm
"""

from Counter.Intersection import ImgCX, ImageWidth, ImageHeight
import numpy as np
import sys
from colorama import Fore

OIS_size = [ImageWidth, ImageHeight]
new_img_size = min(OIS_size)
img_center = ImgCX

 
def calculate_angle(V):
    vector = V.copy()
    center = np.array([new_img_size/2, -new_img_size/2])
    vector[1] = - vector[1] 
    vector1 = vector - center
    vector2 = np.array([0, new_img_size/2])
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(vector1, vector2)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def rotate_coordinate(C, angle_deg):
    coord = C.copy()
    coord[1] = - coord[1]
    center = np.array([new_img_size/2, -new_img_size/2])
    angle_rad = np.radians(angle_deg)
    coord = coord - center
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_coord = np.dot(rotation_matrix, coord)
    rotated_coord = rotated_coord + center
    rotated_coord[0] =  round(rotated_coord[0])
    rotated_coord[1] = - round(rotated_coord[1])
    return rotated_coord


def NMS(bboxes, threshold):
    
    def calculate_iou(box1, box2, threshold):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
    
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
         
        iou = intersection / (area1 + area2 - intersection)
        if iou < threshold and intersection/min(area1, area2) > 0.9:
            iou = threshold + 0.1
            
        return iou

    # Sort bounding boxes by confidence in descending order
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # Initialize list of selected bounding boxes
    selected_boxes = []

    # Iterate through all bounding boxes
    while len(bboxes) > 0:
        # Select the bounding box with highest confidence
        box = bboxes.pop(0)

        # Add the box to the list of selected boxes
        selected_boxes.append(box)

        # Compute the IoU (Intersection over Union) between the selected box and all other boxes
        ious = [calculate_iou(box, b, threshold) for b in bboxes]

        # Remove boxes that overlap significantly with the selected box
        for i in range(len(bboxes)):
            if ious[i] > threshold:
                bboxes[i] = None

        bboxes = [b for b in bboxes if b is not None]

    return selected_boxes

    
def PrintCounts(VD):
    total = 0
    line = ""
    for i in range(1, 5):
        for j in range(1, 5):
            value = VD.Counts[i-1][j-1]
            letter = f"{Fore.WHITE}{i}{j}:{Fore.YELLOW}{value}\t" # Corrected the line variable
            line += letter
            total += value
    
    line += f"Total:{Fore.CYAN}{total}"
    if VD.TrainingMode == True:
        line += f"  {Fore.WHITE}Data collection progress for training:{Fore.RED}{VD.CollectData.progress} %"
    sys.stdout.write("\r")        
    sys.stdout.write(line) 
    sys.stdout.flush()