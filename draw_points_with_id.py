import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
ids = ['A', 'B', 'C', 'D', 'E']

# 创建散点图
plt.scatter(x, y)

# 添加散点ID
for i, id in enumerate(ids):
    plt.annotate(id, (x[i], y[i]))

# 显示图形
plt.show()
