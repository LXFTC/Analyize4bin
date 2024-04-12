import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image

#定义一个全局列表，用于保存circle的坐标
global_circle_list = []

#定义一个全局列表，用于保存蓝光下液滴的信息，包括圆心位置，半径，亮度值
global_blue_list = []

#定义一个全局列表，用于保存绿光下液滴的信息，包括圆心位置，半径，亮度值
global_green_list = []

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


#开运算函数
def opening(img):
    original_image = img.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)#kernel也可以选择椭圆或者矩阵或者十字
    kernel_opening = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening, 3)#去毛刺
    cv2.imwrite("opening.tiff",opening)


#sobel算子边缘检测
def sobel_edge_detection(img):
    # original_image = img.copy()
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_xedge = cv2.Sobel(img,cv2.CV_16S,0,1,3)
    sobel_yedge = cv2.Sobel(img,cv2.CV_16S,1,0,3)
    absx = cv2.convertScaleAbs(sobel_xedge)
    absy = cv2.convertScaleAbs(sobel_yedge)
    sobel_edge_image = cv2.addWeighted(absx,0.5,absy,0.5,0)
     
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

    cv2.imwrite("mask.tiff",mask)
    cv2.imwrite("opening_mask.tiff",opening_mask)
    print(f"mask:{mask.dtype}_{mask.shape}")


#根据sobel算子得到的mask做圆形检测，并将检测到的圆心以及半径存储到全局列表
def draw_mask_circle2BG ():
    opening_mask = cv2.imread("opening_mask.tiff")
    
    opening_mask_G = cv2.cvtColor(opening_mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("opening_mask_G.tiff",opening_mask_G)
    print (f"mask_G数据种类{opening_mask_G.dtype}_{opening_mask_G.shape}")
    #对mask_G做圆形检测
    circles = cv2.HoughCircles(opening_mask_G, cv2.HOUGH_GRADIENT, 1, 15, param1=50, param2=1, minRadius=3, maxRadius=10)
    print("检测到的圆的个数：", circles.shape[1]) 

    #将圆心以及圆的半径等信息存在全局变量中
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            global_circle_list.append((i[0], i[1],i[2]))
            # print(f"圆心坐标：{i[0]},{i[1]}")


#在输入图像上绘制亮度文本以及圆形
def draw_pixel_at_circle_center(img,color_light, picnum):
    height, width = img.shape
    bgr_image = img.copy()
    bgr_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 遍历圆心位置的列表
    for center in global_circle_list:
        if 0 <= center[0] < width and 0 <= center[1] < height:
            # 获取圆心处的像素值
            px_value = img[center[1], center[0]]
        if color_light == "blue":
            # 将圆心处的像素值保存到全局列表
            global_blue_list.append((center[0], center[1], center[2], int (px_value*10-200)))
        elif color_light == "green":
            # 将圆心处的像素值保存到全局列表
            global_green_list.append((center[0], center[1], center[2], int (px_value*10-200)))
        # 在圆心处绘制像素值
        # cv2.putText(img, str(px_value/255*65535), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 0, 0), 1, cv2.LINE_AA)
        bgr_image = cv2.putText(bgr_image, str(px_value*10-200), (center[0]-5, center[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.3, (255, 0, 0), 1)

        # 在像素值上方绘制圆心坐标
        # bgr_image = cv2.putText(bgr_image, str(f"{center[1]}_{center[0]}"), (center[0]-5, center[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        # 在圆心处绘制圆
        cv2.circle(bgr_image, (center[0], center[1]), center[2], (0, 0, 255), 1)
    
    cv2.imwrite(f"{picnum}_{color_light}_{len(global_circle_list)}.tiff",bgr_image)
    # print (f"global_blue_list:{global_blue_list}")
    # print(f"global_green_list:{global_green_list}")


#根据global_blue_list和global_green_list生成json文件
def generate_json_file(color_of_light):
    if color_of_light == "blue":
        json_file_name = "blue.json"
        json_data = {}
        json_data["circles"] = []
        for data in global_blue_list:
            # 定义json文件中每个数据项的内容
            data_content = {"id":f"{data[0]}_{data[1]}", "x": data[0], "y": data[1], "radius": data[2], "FAM": data[3],"islive":"true"}
         # 将uint16类型的数据转换为int类型
            if isinstance(data[0], np.uint16):
                data_content["x"] = int(data[0])
            if isinstance(data[1], np.uint16):
                data_content["y"] = int(data[1])
            if isinstance(data[2], np.uint16):
                data_content["radius"] = int(data[2])
            if isinstance(data[3], np.uint16):
                data_content["FAM"] = int(data[3])
            
            # 将数据项内容添加到json文件中
            json_data["circles"].append(data_content)  # 这里应该使用json_data["circles"]而不是circles
                
        json_data["lightid"] = "1"
        # 保存json文件
        try:
            with open(json_file_name, "w") as f:
                    
                f.write(json.dumps(json_data, indent=4))
                print(f"已保存{color_of_light}光下液滴的json文件：{json_file_name}")
        except PermissionError:
            print("没有足够的权限来创建或写入文件")  


    if color_of_light == "green":
        json_file_name = "green.json"
        json_data = {}
        json_data["circles"] = []
        for data in global_green_list:
            # 定义json文件中每个数据项的内容
            data_content = {"id":f"{data[0]}_{data[1]}", "x": data[0], "y": data[1], "radius": data[2], "HEX": data[3],"islive":"true"}
            # 将uint16类型的数据转换为int类型
            if isinstance(data[0], np.uint16):
                data_content["x"] = int(data[0])
            if isinstance(data[1], np.uint16):
                data_content["y"] = int(data[1])
            if isinstance(data[2], np.uint16):
                data_content["radius"] = int(data[2])
            if isinstance(data[3], np.uint16):
                data_content["HEX"] = int(data[3])
                
            # 将数据项内容添加到json文件中
            json_data["circles"].append(data_content)  # 这里应该使用json_data["circles"]而不是circles
                    
        json_data["lightid"] = "1"
        # 保存json文件
        try:
            with open(json_file_name, "w") as f:
                    
                f.write(json.dumps(json_data, indent=4))
                print(f"已保存{color_of_light}光下液滴的json文件：{json_file_name}")
        except PermissionError:
            print("没有足够的权限来创建或写入文件")  

#合并blue.json和green.json文件
def merge_json(picnum):
    # 加载green.json文件
    with open('green.json', 'r', encoding='utf-8') as file:
        green_data = json.load(file)

    # 加载blue.json文件
    with open('blue.json', 'r', encoding='utf-8') as file:
        blue_data = json.load(file)

    # 创建一个字典，以便快速通过id查找blue中的圆圈
    blue_circles_dict = {circle['id']: circle for circle in blue_data['circles']}
    # 合并圈对象
    merged_circles = []
    for green_circle in green_data['circles']:
        circle_id = green_circle['id']
        if circle_id in blue_circles_dict:
            merged_circle = green_circle.copy()  # 先复制green中的圈对象
            merged_circle.update(blue_circles_dict[circle_id])  # 用blue中的圈对象更新，确保FAM值加入
            merged_circles.append(merged_circle)
        else:
            merged_circles.append(green_circle)  # 如果在blue中找不到相应id的圈对象，直接添加green中的对象

    # 准备最终的合并数据
    merged_data = {
        "circles": merged_circles,
        "lightid": green_data['lightid']  # lightid在两个文件中都是相同的
    }

    # 将合并后的数据写入新的JSON文件
    with open(f'通道{picnum}.json', 'w', encoding='utf-8') as file:
        json.dump(merged_data, file, indent=4, ensure_ascii=False)

    #释放全局列表对于内存的占用
    global_circle_list.clear()
    global_blue_list.clear()
    global_green_list.clear()

     


    
        
   
    






