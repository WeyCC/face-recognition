from matplotlib import pyplot as plt
from sklearn import datasets

# step1:调用人脸数据集
faces = datasets.fetch_olivetti_faces('./')  # 400张人脸图片

i = 0

# step2:绘制人脸图像
plt.figure(figsize=(20, 20))  # 画布设置为20*20

for img in faces.images:
    # 20行，20列，第三个参数代表第几个子图
    plt.subplot(20, 20, i + 1)
    plt.imshow(img)

    # 不显示横纵坐标，单纯显示人脸图像
    plt.xticks([])
    plt.yticks([])

    i = i + 1

plt.show()
