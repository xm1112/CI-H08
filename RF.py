
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# 决策树可视化
from sklearn import tree
import graphviz
import pydotplus
from sklearn.metrics import *
from scoretotal import *
from binary import *
# 忽略警告
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("D:/workspace/RF/LAST/0207/90%/A2/v6_train_D1.csv",encoding='unicode_escape',  error_bad_lines=False)
test = pd.read_csv("D:/workspace/RF/LAST/0207/90%/A2/v6_test_D1.csv", encoding='unicode_escape', error_bad_lines=False)

list_train = train.values.tolist()
list_test = test.values.tolist()

train_Col = np.array(list_train)
test_Col = np.array(list_test)

feature_train = train_Col[:,4:24]
feature_test = test_Col[:,4:24]
target_train = train_Col[:,25]
target_test = test_Col[:,25]

# RF模型
dt_model = RandomForestClassifier(n_estimators = 76,criterion='gini', max_depth=10, max_features =5,min_samples_leaf=1, random_state=445)   # random_state=123 保持预测结果一致
dt_model = dt_model.fit(feature_train, target_train)        #训练
predict_results = dt_model.predict(feature_test)            #预测
# print(predict_results)

# 特征重要性排序
importance = dt_model.feature_importances_
print(importance)  # [0.16178792 0.33465241 0.22366101 0.06106394 0.09598642 0.12284831]
print("特征重要性:")
print(sorted(zip(map(lambda x:round(x,23),importance),train.columns[4:26])))
print("测试集上准确率：", accuracy_score(target_test, predict_results)) #测试集的准确率  (y_true, y_pre)


############计算命中率和虚警率###########################
target_test_arr = np.array(target_test, dtype=int) #数组中字符串转成数值
# print(target_test_arr)
predict_arr = np.array(predict_results, dtype=int)
# print(predict_arr)

# 计算二分类结果
hits, misses, falsealarms, correctnegatives = prep_clf(target_test_arr,predict_arr,2)
print(hits, misses, falsealarms, correctnegatives)
hits, misses, falsealarms, correctnegatives = prep_clf(target_test_arr,predict_arr,3)
print(hits, misses, falsealarms, correctnegatives)

out_l1_pod = POD(target_test_arr, predict_arr, threshold=0)
print("out_l1_pod:", out_l1_pod)
out_l2_pod = POD(target_test_arr, predict_arr, threshold=2)
print("out_l2_pod:", out_l2_pod)
out_l3_pod = POD(target_test_arr, predict_arr, threshold=3)
print("out_l3_pod:", out_l3_pod)

out_l1_FAR = FAR(target_test_arr, predict_arr, threshold=0)
print("out_l1_FAR:", out_l1_FAR)
out_l2_FAR = FAR(target_test_arr, predict_arr, threshold=2)
print("out_l2_FAR:", out_l2_FAR)
out_l3_FAR = FAR(target_test_arr, predict_arr, threshold=3)
print("out_l3_FAR:", out_l3_FAR)


out_l1_CSI = CSI(target_test_arr, predict_arr, threshold=0)
print("out_l1_CS1:", out_l1_CSI)
out_l2_CSI = CSI(target_test_arr, predict_arr, threshold=2)
print("out_l2_CSI:", out_l2_CSI)
out_l3_CSI = CSI(target_test_arr, predict_arr, threshold=3)
print("out_l3_CSI:", out_l3_CSI)


############预测结果概率###########################
# 预测结果的概率
predict_proba = dt_model.predict_proba(feature_test)  # #这个是得分,每个分类器的得分，取最大得分对应的类。
# print(predict_proba[:,-1])

list = []
for i in range(len(predict_results)):
    perc = predict_proba[i]      # 预测结果概率
    # print(type(perc))          # <class 'numpy.ndarray'>
    p = np.insert(perc,0,predict_results[i]) # 插入预测值
    p = np.insert(p,0,target_test[i])   # 第一行插入真值
    list.append(p)
# print(list)                # (真值,预测值,0的概率,1的概率,2的概率)


# # 输出到表格
# dataFrame = pd.DataFrame(list, index = None,columns=["true_result","predict_result",'0', '2',"3"],) # 说明行和列的索引名
# path = "D:/workspace/RF/LAST/A8/"
# with pd.ExcelWriter( path +'D1.xlsx') as writer: # 写入数据
#     dataFrame.to_excel(writer, sheet_name='page1', float_format='%.6f')
#

# 提取真值为 3 的概率信息
y_label = []
y_score = []
for i in range(len(list)):
    if list[i][0] == 3:
        label = [0,0,1]   # list类型  制作标签
        score = predict_proba[i]
        y_label.append(label)
        y_score.append(score)
# print(y_label,y_score)

y_label = np.array(y_label).ravel()
y_score = np.array(y_score).ravel()
# print(y_label,y_score)

"""
# ROC曲线
fpr, tpr, thresholds = roc_curve(y_label,y_score)   # 输入真值
# print(fpr)
aucval = auc(fpr,tpr)                               # 计算auc的取值
print("auc面积：",aucval)

plt.plot(fpr, tpr,"r",linewidth = 3)
plt.grid()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("V4-1:1:1")
plt.text(0.15,0.9,"AUC = "+str(round(aucval,4)))
plt.show()
"""
