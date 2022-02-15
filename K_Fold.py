import glob
import warnings

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

PICTURE_PATH = "./data/orl_faces"

all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签


def get_picture():
    label = 1
    # 读取所有图片并一维化
    while (label <= 20):
        for name in glob.glob(PICTURE_PATH + "\\s" + str(label) + "\\*.pgm"):
            img = Image.open(name)
            # img.getdata()
            # np.array(img).reshape(1, 92*112)
            all_data_set.append(list(img.getdata()))
            all_data_label.append(label)
        label += 1


get_picture()

x = np.array(all_data_set)
y = np.array(all_data_label)

# step2:划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# step3:计算评估指标
def calculate_metric(gt, pred):
    pred[pred > 0.5] = 1
    pred[pred < 1] = 0
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    accuracy = (TP + TN) / float(TP + TN + FP + FN)
    specificity = TN / float(TN + FP)
    recall = TP / float(TP + FN)

    return accuracy, specificity, recall


# step4:进行交叉验证，4、5、6、7、8折
for fold in range(4, 9):
    print(f'[Fold:{fold}]')
    # 构建KFold对象用对切分数据集，为之后进行K折验证
    kf = KFold(n_splits=fold, random_state=0, shuffle=True)

    # 计算评估指标
    accuracy, specificity, recall = 0.0, 0.0, 0.0
    for k, (train, test) in enumerate(kf.split(x_train, y_train)):
        x_ = x_train[train]
        y_ = y_train[train]

        x__ = x_train[test]
        y__ = y_train[test]

        model = SVC()
        model.fit(x_, y_)
        a_ = model.score(x_test, y_test)
        b_ = precision_score(y_test, model.predict(x_test), average='macro')
        c_ = recall_score(y_test, model.predict(x_test), average='macro')
        accuracy += a_
        specificity += b_
        recall += c_

    print('准确率:', accuracy / fold)
    print('特异性:', specificity / fold)
    print('召回率:', recall / fold)

