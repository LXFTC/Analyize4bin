import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image
import split_circle
import threading 
 
Wimg = cv2.imread('sobel_tiff\\0513170420_BL1_07.tiff', cv2.IMREAD_ANYDEPTH)
Bimg = cv2.imread('sobel_tiff\\0513170602_FAM1_08.tiff', cv2.IMREAD_ANYDEPTH)
Gimg = cv2.imread('sobel_tiff\\0513170900_HEX1_09.tiff', cv2.IMREAD_ANYDEPTH)

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


#对Wimg做sobel算子边缘检测
split_circle.sobel_edge_detection(Wimg)
split_circle.connected_component_process("sobel_edge_image.tiff")
# lock = threading.Lock()

# def thread_Func(img,color_of_light):
#     lock.acquire()
#     split_circle.draw_mask_circle2BG (img)
#     split_circle.draw_pixel_at_circle_center(img,color_of_light)
#     split_circle.generate_json_file(color_of_light)
#     lock.release()
    


# Thread1 = threading.Thread(target=thread_Func, args=(Bimg,'blue'))
# Thread1.start()
# Thread1.join() 

# Thread2 = threading.Thread(target=thread_Func, args=(Gimg,'green'))
# Thread2.start()
# Thread2.join()    

split_circle.draw_mask_circle2BG (Gimg)
split_circle.draw_pixel_at_circle_center(Gimg,'green')   
split_circle.generate_json_file('green')


split_circle.draw_pixel_at_circle_center(Bimg,'blue')   
split_circle.generate_json_file('blue')

# split_circle.merge_json('通道1.json')




# split_circle.generate_json_file('blue')
# split_circle.generate_json_file('green')

# split_circle.merge_json('blue.json','green.json')





