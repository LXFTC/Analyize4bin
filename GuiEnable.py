# 调用main.py中的函数进行图片分析，并显示进度条
import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QProgressBar,QFileDialog
from PySide6.QtCore import QTimer
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image
import split_circle


class FileSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('小图分析')

        # 创建水平布局并添加按钮
        h_layout = QHBoxLayout()
        self.btn_select = QPushButton('选择文件夹', self)
        self.btn_select.clicked.connect(self.on_select_folder)
        self.btn_analyze = QPushButton('开始分析', self)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.on_analyze)
        h_layout.addWidget(self.btn_select)
        h_layout.addWidget(self.btn_analyze)

        # 创建垂直布局并添加水平布局、文本编辑组件和进度条
        v_layout = QVBoxLayout()
        self.text_edit = QTextEdit(self)
        self.progress_bar = QProgressBar(self)
        # self.progress_bar.setRange(0, 16)  # 设置进度条的范围为0到16
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.text_edit)
        v_layout.addWidget(self.progress_bar)

        # 将垂直布局设置为窗口部件的布局
        self.setLayout(v_layout)

        # 创建定时器
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_progress_from_main)

        
    # 连接按钮点击事件
    def on_select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            self.selected_path = folder_path
            self.text_edit.append(f'选择的文件夹路径: {folder_path}')
        self.btn_analyze.setEnabled(True)

    def on_analyze(self):
        self.btn_analyze.setEnabled(False)  # 禁用开始分析按钮
        self.text_edit.append('开始分析...')
        self.progress_bar.setValue(0)  # 重置进度条
        folder_path = file_selector.selected_path  # 获取文件夹路径
        #在路径下新建文件夹
        # result_picture_folder_name = '图像标注结果'
        # re_picture_path = os.mkdir(os.path.join(folder_path, result_picture_folder_name),exit_ok=True)
        # result_json_folder_name = 'json文件'
        # re_josn_path = os.mkdir(os.path.join(folder_path, result_json_folder_name),exit_ok=True)
        # # 调用main.py中的函数进行图片分析，并显示进度条
        start = time.time()

        # 获取文件夹下所有的TIFF文件
        files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]

        # 确保文件按照文件名排序，如果文件名包含数字，这将保证顺序正确
        files.sort()

        # 分组，每三个一个组
        grouped_files = [files[i:i+3] for i in range(0, len(files), 3)]

        # 将每组的文件名加入到一个新的总列表中
        folder_names = []
        self.picnum = 0
        for group in grouped_files:
            
            Wimg_path = folder_path + '\\' + os.path.splitext(group[0])[0] + '.tiff'
            Bimg_path = folder_path + '\\' + os.path.splitext(group[1])[0] + '.tiff'
            Gimg_path = folder_path + '\\' + os.path.splitext(group[2])[0] + '.tiff'
            
            Wimg = cv2.imread(Wimg_path, cv2.IMREAD_ANYDEPTH)
            Bimg = cv2.imread(Bimg_path, cv2.IMREAD_ANYDEPTH)
            Gimg = cv2.imread(Gimg_path, cv2.IMREAD_ANYDEPTH)

            # 图像增强
            Wimg = split_circle.enhance_contrast(Wimg, 10, 3000)
            Bimg = split_circle.enhance_contrast(Bimg, 50, 4000)#30，4000
            Gimg = split_circle.enhance_contrast(Gimg, 70, 3000)

            #对Wimg做sobel算子边缘检测
            sobel_edge_image = split_circle.sobel_edge_detection(Wimg)
            sobel_image_name = "sobel_edge_image.tiff"
            cv2.imwrite(os.path.join(re_picture_path, sobel_image_name), sobel_edge_image)
            opening_mask = split_circle.connected_component_process(os.path.join(re_picture_path, sobel_image_name))
            opening_mask_name = "opening_mask.tiff"
            cv2.imwrite(os.path.join(re_picture_path, opening_mask_name), opening_mask)
            split_circle.draw_mask_circle2BG (os.path.join(re_picture_path, opening_mask_name))

            split_circle.draw_pixel_at_circle_center(Bimg,'blue',self.picnum)
            split_circle.generate_json_file('blue')

            split_circle.draw_pixel_at_circle_center(Gimg,'green',self.picnum)
            split_circle.generate_json_file('green')

            self.picnum += 1
            split_circle.merge_json(self.picnum)
            self.text_edit.append(f'已完成第{self.picnum}张图片的分析。')
            self.progress_bar.setValue(int((self.picnum / 16) * 100))  # 更新进度条

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
        self.text_edit.append(f'分析完成，用时{int(end - start)}秒。')



    # def start_progress(self):
    #     self.progress = 0
    #     self.progress_bar.setValue(0)
    #     self.timer.start(100)  # 以100毫秒的间隔更新进度条

    # def stop_progress(self):
    #     self.timer.stop()  # 停止进度条更新



if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    file_selector = FileSelector()
    file_selector.show()
    # file_selector.start_progress()  # 开始进度条更新
    sys.exit(app.exec())




