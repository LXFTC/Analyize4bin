import json
import numpy as np
import matplotlib.pyplot as plt

# 创建一个包含16个子图的画布
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

# 循环读取16个JSON文件
for i in range(1, 17):
    file_name = f'通道{i}.json'  # 构建文件名
    x_data = []
    y_data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for circle in data['circles']:
            x = circle['HEX']
            y = circle['FAM']
            x_data.append(x)
            y_data.append(y)
        print(f'读取文件 {file_name} 内容成功！')

    # 将数据转换为NumPy数组
    coordinates = np.column_stack((x_data, y_data))

    # 获取当前子图并绘制scatter plot
    ax = axs[(i-1)//4, (i-1)%4]  # 确定子图位置
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=1)
    ax.set_title(f'Scatter Plot {i}')  # 设置子图标题

# 显示图像
plt.show()
