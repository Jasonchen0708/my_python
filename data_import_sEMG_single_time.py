import numpy as np
import os,glob
from scipy.io import loadmat, savemat
from sklearn import svm
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler  # 导入标准化数据
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
path = '/document/chenyan/Downloads/model_data/qu/'
# path = '/home/chenyan/Downloads/model_data/qu/'
files = glob.glob(os.path.join(path, "*.mat"))
filenames = []
data =np.zeros([1, 25])
label =np.zeros([1, 1])
for dir_t in files:
    filenames.append(loadmat(dir_t))
for i in range(len(filenames)):
    trans = np.array(filenames[i]['test'])
    trans2 = np.array(filenames[i]['test_label'])
    data = np.vstack((data, trans))
    label = np.hstack((label, trans2))
data = data[1:, :]
label = label[:, 1:]
savemat('/home/chenyan/Downloads/all_qu_test_data.mat', {'test_data': data, 'label': label})
# 4.生成结果报告
# print('The Accuracy of Linear SVC is %f' % model.score(X_test, y_test))  # 使用自带的模型评估函数进行准确性评测
# print(classification_report(y_test, y_predict))
# print("训练集：",model.score(X_train,y_train))
# print("测试集：",model.score(X_test,y_test))

