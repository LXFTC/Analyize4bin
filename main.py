import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image
import split_circle
from threading import Thread, Lock
 
Wimg = cv2.imread('sobel_tiff\\0513173604_BL1_10.tiff', cv2.IMREAD_ANYDEPTH)
Bimg = cv2.imread('sobel_tiff\\0513173810_FAM1_11.tiff', cv2.IMREAD_ANYDEPTH)
Gimg = cv2.imread('sobel_tiff\\0513174113_HEX1_12.tiff', cv2.IMREAD_ANYDEPTH)

print(f"Wimg读进来的格式:{Wimg.dtype}+{Wimg.shape}")
print(f"Bimg读进来的格式:{Bimg.dtype}+{Bimg.shape}")
print(f"Gimg读进来的格式:{Gimg.dtype}+{Gimg.shape}")

# 图像增强
Wimg = split_circle.enhance_contrast(Wimg, 10, 3000)
Bimg = split_circle.enhance_contrast(Bimg, 50, 4000)#30，4000
Gimg = split_circle.enhance_contrast(Gimg, 80, 3000)

print(f"Wimg转换后的格式:{Wimg.dtype}+{Wimg.shape}")
print(f"Bimg转换后的格式:{Bimg.dtype}+{Bimg.shape}")
print(f"Gimg转换后的格式:{Gimg.dtype}+{Gimg.shape}")

# cv2.imshow("Bimg",Bimg)
# cv2.imshow("Gimg",Gimg)
# cv2.imshow("Wimg",Wimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#对Wimg做sobel算子边缘检测
split_circle.sobel_edge_detection(Wimg)
split_circle.connected_component_process("sobel_edge_image.tiff")

def thread_Func(img,color_of_light):
    split_circle.draw_mask_circle2BG (img)
    
    split_circle.draw_pixel_at_circle_center(img,color_of_light)



Thread1 = Thread(target=thread_Func, args=(Bimg,'blue'))
Thread2 = Thread(target=thread_Func, args=(Gimg,'green'))

Thread1.start()    
Thread2.start()

Thread1.join()    
Thread2.join()









