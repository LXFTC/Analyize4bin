import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image
import split_circle
import threading 
 
# # 文件夹路径
# folder_path = 'D:\\opencv-python\\sobel_tiff'#D:\\opencv-python\\sobel_tiff
# start = time.time()
# # 获取文件夹下所有的TIFF文件
# files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]

# # 确保文件按照文件名排序，如果文件名包含数字，这将保证顺序正确
# files.sort()

# # 分组，每三个一个组
# grouped_files = [files[i:i+3] for i in range(0, len(files), 3)]

# # 将每组的文件名加入到一个新的总列表中
# folder_names = []
# picnum = 0
# for group in grouped_files:
    
#     Wimg_path = folder_path + '\\' + os.path.splitext(group[0])[0] + '.tiff'
#     Bimg_path = folder_path + '\\' + os.path.splitext(group[1])[0] + '.tiff'
#     Gimg_path = folder_path + '\\' + os.path.splitext(group[2])[0] + '.tiff'
    
Wimg = cv2.imread(r'D:\\opencv-python\\sobel_tiff\\0513160972_BL1_01.tiff', cv2.IMREAD_ANYDEPTH)
Bimg = cv2.imread(r'D:\\opencv-python\\sobel_tiff\\0513161161_FAM1_02.tiff', cv2.IMREAD_ANYDEPTH)
Gimg = cv2.imread(r'D:\\opencv-python\\sobel_tiff\\0513161433_HEX1_03.tiff', cv2.IMREAD_ANYDEPTH)

    # 图像增强
Wimg = split_circle.enhance_contrast(Wimg, 10, 3000)
Bimg = split_circle.enhance_contrast(Bimg, 50, 4000)#30，4000
Gimg = split_circle.enhance_contrast(Gimg, 70, 3000)

    #对Wimg做sobel算子边缘检测
split_circle.sobel_edge_detection(Wimg)
split_circle.connected_component_process("sobel_edge_image.tiff")
split_circle.draw_mask_circle2BG ("sobel_edge_image.tiff")

split_circle.draw_pixel_at_circle_center(Bimg,'blue',0)
split_circle.generate_json_file('blue')

split_circle.draw_pixel_at_circle_center(Gimg,'green',0)
split_circle.generate_json_file('green')

    # def thread_Func(img,color_of_light,picnum):
    #     split_circle.draw_pixel_at_circle_center(img,color_of_light,picnum)
    #     split_circle.generate_json_file(color_of_light)
        

    # Thread1 = threading.Thread(target=thread_Func, args=(Bimg,'blue'))
    # Thread1.start()
    # Thread1.join() 

    # Thread2 = threading.Thread(target=thread_Func, args=(Gimg,'green'))
    # Thread2.start()
    # Thread2.join()    

# split_circle.merge_json(picnum)

    # picnum += 1

#删除中间生成的图片文件以及中间的json文件
# try:
#     os.remove("blue.json")
#     os.remove("green.json")
#     os.remove("sobel_edge_image.tiff")
#     os.remove("mask.tiff")
#     os.remove("opening_mask_G.tiff")
#     os.remove("opening_mask.tiff")
#     print(f"文件已被成功删除。")
# except FileNotFoundError:
#     print(f"文件不存在，无法删除。")
    
end = time.time()
# print (f"程序运行时间：{int (end - start)} s")






