import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB  # 导入朴素贝叶斯分类器
from sklearn import metrics  # 从sklearn工具包导入metrics模块
from sklearn.metrics import accuracy_score, mean_squared_error

# 加载数据集
file_path = r".\01.csv"
df = pd.read_csv(file_path)

X = df.drop('NObeyesdad', axis=1)  # 特征
y = df['NObeyesdad']  # 目标变量

# 将标签编码为0和1
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
le = LabelEncoder()
y = le.fit_transform(y)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练朴素贝叶斯模型
model = GaussianNB()  # 适用于连续数据的朴素贝叶斯
model.fit(X_train, y_train)

# 预测数据集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("分类准确率:", accuracy)

# 将编码后的预测和测试集标签还原为原始类别标签
y_test_original = le.inverse_transform(y_test)
y_pred_original = le.inverse_transform(y_pred)

# 打印分类报告
print(metrics.classification_report(y_test_original, y_pred_original, digits=4))

# 打印混淆矩阵
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# 计算MSE指标
mse = mean_squared_error(y_test, y_pred)  # 计算MSE指标
print("测试集上的MSE指标: %.4f" % mse)  # 输出MSE指标
