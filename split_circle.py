import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image

Wimg = cv2.imread("D:\\opencv-python\\sobel_tiff\\0513160972_BL1_01.tiff")
Bimg = cv2.imread("D:\\opencv-python\\sobel_tiff\\0513161161_FAM1_02.tiff",cv2.IMREAD_UNCHANGED)
Gimg = cv2.imread("D:\\opencv-python\\sobel_tiff\\0513161433_HEX1_03.tiff",cv2.IMREAD_UNCHANGED)
# plt.imshow(Wimg)
# plt.imshow(Bimg)
# plt.imshow(Gimg)
# plt.show()

print(f"Wimg读进来的格式:{Wimg.dtype}+{Wimg.shape}")
print(f"Bimg读进来的格式:{Bimg.dtype}+{Bimg.shape}")
print(f"Gimg读进来的格式:{Gimg.dtype}+{Gimg.shape}")

def opening(img):
    original_image = img.copy()
    # re_image = cv2.resize(image,(400,300))
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), dtype=np.uint8)#kernel也可以选择椭圆或者矩阵或者十字
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 3)#去毛刺
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 3)#去毛刺
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 3)#去毛刺
    # ss = np.hstack((gray_img, opening))
    # plt.imshow(opening)
    # plt.show()
    cv2.imwrite("opening.tiff",opening)
#     cv2.imshow("opening_ss",opening)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def sobel_edge_detection(img):

    original_image = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_xedge = cv2.Sobel(gray_img,cv2.CV_16S,0,1,3)
    sobel_yedge = cv2.Sobel(gray_img,cv2.CV_16S,1,0,3)
    absx = cv2.convertScaleAbs(sobel_xedge)
    absy = cv2.convertScaleAbs(sobel_yedge)
    sobel_edge_image = cv2.addWeighted(absx,0.5,absy,0.5,0)
    # sobel_edge_image_T = cv2.threshold(sobel_edge_image, 3, 250, cv2.THRESH_BINARY)
    # plt.imshow(sobel_edge_image)
    # plt.show()
    cv2.imwrite("sobel_edge_image.tiff",sobel_edge_image)
    print(f"sobel_edge_img:{sobel_edge_image.dtype}+{sobel_edge_image.shape}")
    # cv2.imshow("sobel_edge_image_T",sobel_edge_image_T)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#极大连通域处理
def connected_component_process(img):
    img = cv2.imread(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_T = cv2.threshold(gray_image, 7, 255, cv2.THRESH_BINARY)[1]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_T, connectivity=8, ltype=cv2.CV_32S)

    max_label = 0
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_label = i
            max_area = stats[i, cv2.CC_STAT_AREA]

    # 创建一个掩码来显示极大连通区域
    mask = np.zeros_like(img)
    mask[labels == max_label] = 255
    # 显示掩码
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("mask.tiff",mask)
    
    print(f"mask:{mask.dtype}_{mask.shape}")

def draw_mask_circle2BG (img):
    mask = cv2.imread("mask.tiff")
    # mask = mask.astype(np.uint16)
    mask_G = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("mask_G.tiff",mask_G)
    print (f"mask_G数据种类{mask_G.dtype}_{mask_G.shape}")
    #对mask_G做圆形检测
    circles = cv2.HoughCircles(mask_G, cv2.HOUGH_GRADIENT, 1, 18, param1=50, param2=1, minRadius=3, maxRadius=15)
    # print("检测到的圆的个数：", circles.shape[0]) 
    # #对mask创建掩膜
    # mask_circle = np.zeros_like(img)
    mask_circle = np.zeros((752, 1028))

    Bimg = cv2.imread("D:\\opencv-python\\sobel_tiff\\0513161161_FAM1_02.tiff")

    #绘制实心圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask_circle, (i[0], i[1]), i[2], (255, 255, 255), 1)
       
    cv2.imshow("mask_circle",mask_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#sobel边缘检测，输出为sobel_edge_image
sobel_edge_detection(Wimg)
connected_component_process("sobel_edge_image.tiff")
draw_mask_circle2BG (Bimg)


