import glob
import warnings

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
x = MinMaxScaler().fit_transform(x)
# step2:划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# step3:计算评估指标
def calculate_metric(gt, pred):
    pred[pred > 0.5] = 1
    pred[pred < 0.5] = 0
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    accuracy = (TP + TN) / float(TP + TN + FP + FN)
    specificity = TN / float(TN + FP)
    recall= TP / float(TP + FN)

    return accuracy,specificity, recall


model_names = ['LinearModel', 'DecisionTree', 'SVM', 'RandomForest', 'DNN']  # 算法名称

# step4:迭代4种算法进行验证不同模型在不同子集上的表现
for index, model in enumerate([LogisticRegression(solver='liblinear'),
                               DecisionTreeClassifier(),
                               SVC(kernel='linear', probability=True, max_iter=1000),
                               RandomForestClassifier(),
                               MLPClassifier(max_iter=1000)]):
    print(f'[{model_names[index]}]')
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    specificity = precision_score(y_test, model.predict(x_test), average='macro')
    recall = recall_score(y_test, model.predict(x_test), average='macro')

    print('准确率:', accuracy)
    print('特异性:', specificity)
    print('召回率:', recall)
