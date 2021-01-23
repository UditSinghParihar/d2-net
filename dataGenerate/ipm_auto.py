import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import perspective, Plane, load_camera_params, bilinear_sampler




def ipm_from_parameters(image, xyz, K, RT):
    P = K @ RT
    print(P)
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    print(pixel_coords.shape, 'zz')
    image2 = bilinear_sampler(image, pixel_coords)
    return image2.astype(np.uint8)


def go(filename, cam, seq):
    ################
    # Derived method
    ################
    # Define the plane on the region of interest (road)
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    TARGET_H, TARGET_W = 1024,1024
    plane = Plane(20, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.035)
    # Retrieve camera parameters
    extrinsic, intrinsic = load_camera_params(cam)
    # Apply perspective transformation
    warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic)
    write_name = 'top_view/' + seq + '/' + filename
    cv2.imwrite(write_name, cv2.cvtColor(warped1, cv2.COLOR_BGR2RGB))

    
with open("/home/udit/udit/d2-net/dataGenerate/imagePairsOxford.csv") as f:
    lis = [line.split() for line in f]        
    for i, x in enumerate(lis): 
        filename1, filename2 = x[0].split(',')
        go(filename1.split('/')[-1], 'cameraFront.json', 'front')
        go(filename2.split('/')[-1], 'cameraRear.json', 'rear')
 
