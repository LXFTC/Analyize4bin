import cv2
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image

#定义一个全局列表，用于保存circle的坐标
global_circle_list = []

#图像增亮
def enhance_contrast(img, alpha, beta):
    # 提高对比度
    img = alpha * img + beta
    # 限制像素值范围
    img = np.clip(img, 0, 65535)
    # 转换回uint8类型
    img = (img /65535 * 255).astype('uint8')
    #限制像素值范围
    img = np.clip(img, 0, 255)
    return img


def opening(img):
    original_image = img.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)#kernel也可以选择椭圆或者矩阵或者十字
    kernel_opening = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening, 3)#去毛刺
    cv2.imwrite("opening.tiff",opening)


def sobel_edge_detection(img):

    original_image = img.copy()
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_xedge = cv2.Sobel(img,cv2.CV_16S,0,1,3)
    sobel_yedge = cv2.Sobel(img,cv2.CV_16S,1,0,3)
    absx = cv2.convertScaleAbs(sobel_xedge)
    absy = cv2.convertScaleAbs(sobel_yedge)
    sobel_edge_image = cv2.addWeighted(absx,0.5,absy,0.5,0)
     
    # plt.imshow(sobel_edge_image)
    # plt.show()
    cv2.imwrite("sobel_edge_image.tiff",sobel_edge_image)
    print(f"sobel_edge_img:{sobel_edge_image.dtype}+{sobel_edge_image.shape}")
   

#极大连通域处理
def connected_component_process(img):
    img = cv2.imread(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_T = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)[1]

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

    #对掩码进行开运算
    kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 显示掩码
    cv2.imshow("openging_Mask", opening_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("mask.tiff",mask)
    cv2.imwrite("opening_mask.tiff",opening_mask)
    print(f"mask:{mask.dtype}_{mask.shape}")

def draw_mask_circle2BG (img):
    opening_mask = cv2.imread("opening_mask.tiff")
    # mask = mask.astype(np.uint16)
    opening_mask_G = cv2.cvtColor(opening_mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("opening_mask_G.tiff",opening_mask_G)
    print (f"mask_G数据种类{opening_mask_G.dtype}_{opening_mask_G.shape}")
    #对mask_G做圆形检测
    circles = cv2.HoughCircles(opening_mask_G, cv2.HOUGH_GRADIENT, 1, 15, param1=50, param2=1, minRadius=3, maxRadius=10)
    print("检测到的圆的个数：", circles.shape[1]) 

    
    # #对mask创建掩膜
    # mask_circle = np.zeros_like(img)
    # mask_circle = np.zeros((752, 1028))
    
    #对输入图像做开运算
    kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # enhance_contrast(img, 1.5, 0)
    # adjusted_img = cv2.addWeighted(img, 100, np.zeros(img.shape, img.dtype), 50, 0) #这个函数是可用的
    #增强图像亮度
    # 定义亮度增强参数
    # brightness_factor = 254
    # adjusted_img = np.where((255 - img) < brightness_factor, 255, img * 100) #这个函数是可用的

    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #在Bimg或Gimg上绘制圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), 1)
            #将圆心坐标保存到全局列表
            global_circle_list.append((i[0], i[1],i[2]))
            # print(f"圆心坐标：{i[0]},{i[1]}")

       
    # cv2.imshow("mask_circle",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#将uint8图像转换为uint16图像
def uint8_to_uint16(img):
    # img = img.astype(np.uint16)
    img = img /255 * 65535
    return img

#根据圆心位置的列表坐标得出圆心处的像素值
def get_pixel_value(img, global_circle_list):
    pixel_value_list = []
    for i in global_circle_list:
        pixel_value_list.append(img[i[1], i[0]])
    return pixel_value_list



def draw_pixel_at_circle_center(img,color_of_light):
    height, width = img.shape
    # 遍历圆心位置的列表
    for center in global_circle_list:
        if 0 <= center[0] < width and 0 <= center[1] < height:
            # 获取圆心处的像素值
            px_value = img[center[1], center[0]]

        # 在圆心处绘制像素值
        # cv2.putText(img, str(px_value/255*65535), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(px_value*10), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        # 在像素值上方绘制圆心坐标
        # cv2.putText(img, str(f"{center[1]}_{center[0]}"), (center[0], center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
        # 在圆心处绘制圆
        cv2.circle(img, (center[0], center[1]), center[2], (0, 0, 255), 1)
    # 显示绘制了像素值的图像
    # cv2.imshow('Image with pixel values at circle centers', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"img_with_pixel_values_at_circle_centers_{color_of_light}.tiff",img)







