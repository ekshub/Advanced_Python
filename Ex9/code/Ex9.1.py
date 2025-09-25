import numpy as np
import mindspore
from mindspore import Tensor, nn
from mindspore.dataset import GeneratorDataset
import matplotlib.pyplot as plt
import time
import os

# 定义传感器参数和数据路径
sensor_num = 23  # 传感器数量
horizon = 5  # 预测的时间步数
PV_index = list(range(9))  # PV变量的索引范围
OP_index = list(range(9, 18))  # OP变量的索引范围
DV_index = list(range(18, sensor_num))  # DV变量的索引范围
data_path = 'D:\\VSC_code\\train.csv'  # 数据文件路径
seed = 2024

np.random.seed(seed)
mindspore.set_seed(seed)
# 加载数据
data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=range(1, sensor_num + 1))
print(f'数据形状：{data.shape}，元素类型：{data.dtype}')

# 定义数据生成函数
def generateData(data, X_len, Y_len, sensor_num):
    point_num = data.shape[0]
    sample_num = point_num - X_len - Y_len + 1
    X = np.zeros((sample_num, X_len, sensor_num))
    Y = np.zeros((sample_num, Y_len, sensor_num))
    for i in range(sample_num):
        X[i] = data[i:i + X_len]
        Y[i] = data[i + X_len:i + X_len + Y_len]
    return X, Y

# 生成数据集
X_t2, Y_t2 = generateData(data, 30, horizon, sensor_num)
print(f'任务数据集输入数据形状：{X_t2.shape}，输出数据形状：{Y_t2.shape}')

# 数据集划分函数
def splitData(X, Y):
    N = X.shape[0]
    train_X, train_Y = X[:int(N * 0.6)], Y[:int(N * 0.6)]
    val_X, val_Y = X[int(N * 0.6):int(N * 0.8)], Y[int(N * 0.6):int(N * 0.8)]
    test_X, test_Y = X[int(N * 0.8):], Y[int(N * 0.8):]
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# 划分数据集
train_X_t2, train_Y_t2, val_X_t2, val_Y_t2, test_X_t2, test_Y_t2 = splitData(X_t2, Y_t2)
print(f'训练集样本数：{train_X_t2.shape[0]}，验证集样本数：{val_X_t2.shape[0]}，测试集样本数：{test_X_t2.shape[0]}')

# 自定义数据集类
class MultiTimeSeriesDataset:
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

# 生成MindSpore数据集函数
def generateMindsporeDataset(X, Y, batch_size):
    dataset = MultiTimeSeriesDataset(X.astype(np.float32), Y.astype(np.float32))
    dataset = GeneratorDataset(dataset, column_names=['data', 'label'])
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset

# 创建训练、验证和测试数据集
train_dataset_t2 = generateMindsporeDataset(train_X_t2, train_Y_t2, batch_size=32)
val_dataset_t2 = generateMindsporeDataset(val_X_t2, val_Y_t2, batch_size=32)
test_dataset_t2 = generateMindsporeDataset(test_X_t2, test_Y_t2, batch_size=32)

# 输出数据集样本形状
for data, label in train_dataset_t2.create_tuple_iterator():
    print("对于任务二：")
    print('数据形状：', data.shape, '，数据类型：', data.dtype)
    print('标签形状：', label.shape, '，数据类型：', label.dtype)
    break
class TCN_MLP_with_Bias_Block_More(nn.Cell): #定义TCN_MLP_with_Bias_Block类
    def __init__(self): #构造方法
        super().__init__() #调用父类的构造方法
        #比MULTI_STEP_TCN_MLP类增加一个偏差块
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(28,1), pad_mode='valid')
        )
        #对预测时间步数据的嵌入编码操作
        self.step_embedding = nn.Embedding(horizon, sensor_num)
        #对不同传感器的数据做融合（提取传感器数据间的关联特征）
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )
        #对时间序列做卷积（提取时间点数据间的关联特征）
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        )
        #通过一个卷积层得到最后的预测结果
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid') #使用26*1卷积核，不补边
        
    def construct(self, x, iter_step): #construct方法
        h = self.spatial_mlp(x) #经过spatial_mlp空间处理后，得到的数据h的形状：[batch_size, 30, 23]
        #输入数据x的形状：[batch_size, 30, 23]
        h = x + h #残差连接，将x和h对应元素相加，得到的数据h的形状：[batch_size, 30, 23]
        h = h.unsqueeze(1) #根据卷积操作需要，将3维数据升为4维数据：[batch_size, 1, 30, 23]
        h = self.tcn(h) #经过tcn时间卷积后，得到的数据x的形状：[batch_size, 1, 26, 23]
        y = self.final_conv(h) #通过26*1的卷积操作后，得到的数据y的形状：[batch_size, 1, 1, 23]
        y = y.squeeze(1) #将前面增加的维度去掉，得到的数据y的形状：[batch_size, 1, 23]
        #计算时间步数据的嵌入编码
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor) #step_embedding的形状：[batch_size,1,23]
        
        
        concat_op = mindspore.ops.Concat(axis=1)
        bias_input = concat_op((x, step_embedding,y)) #bias_input的形状：[batch_size,32,23]
        bias_input = bias_input.unsqueeze(1)
        bias_input = self.tcn(bias_input)
        
        
        bias_input = bias_input.squeeze(1)
        
        
        bias_output = self.bias_block(bias_input.unsqueeze(1)) #[batch_size, 1, 1, 23]
        bias_output = bias_output.squeeze(1) # [batch_size, 1, 23]
        #加上偏差块的预测结果
        y = y + bias_output #y的形状：[batch_size, 1, 23]
        return y #返回计算结果
class TCN_MLP_Without_Bias_Block(nn.Cell):
    def __init__(self):
        super().__init__()
        # 移除了 bias_block

        # 时间步嵌入
        self.step_embedding = nn.Embedding(horizon, sensor_num)

        # 空间MLP层
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )

        # 时序卷积层
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), pad_mode='valid'),
        )

        # 最终卷积层
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid')

    def construct(self, x, iter_step):
        h = self.spatial_mlp(x)
        h = x + h  # 残差连接
        h = h.unsqueeze(1)  # [batch_size, 1, seq_len, sensor_num]
        h = self.tcn(h)
        y = self.final_conv(h)
        y = y.squeeze(1)  # [batch_size, 1, sensor_num]

        # 时间步嵌入编码
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor)

        # 在没有 bias_block 的情况下，直接将 y 与 step_embedding 相加
        y = y + step_embedding
        return y
class TCN_MLP_Without_Step_Embedding(nn.Cell):
    def __init__(self): #构造方法
        super().__init__() #调用父类的构造方法
        # 增加偏差块
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(28,1), pad_mode='valid')
        )
        # 移除 step_embedding 部分，不再对预测时间步进行嵌入编码
        # self.step_embedding = nn.Embedding(horizon, sensor_num) # 已移除
        
        # 对不同传感器的数据做融合（提取传感器数据间的关联特征）
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )
        # 对时间序列做卷积（提取时间点数据间的关联特征）
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        )
        # 通过一个卷积层得到最后的预测结果
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid') #使用26*1卷积核，不补边

    def construct(self, x, iter_step): #construct方法
        # 输入数据x的形状：[batch_size, 30, 23]
        h = self.spatial_mlp(x) #经过spatial_mlp空间处理后，h的形状与x相同：[batch_size, 30, 23]
        h = x + h # 残差连接
        # h形状仍为：[batch_size, 30, 23]

        h = h.unsqueeze(1) # 升维为4D：[batch_size, 1, 30, 23]
        h = self.tcn(h) # 经过tcn后，假设输出形状：[batch_size, 1, 26, 23]
        y = self.final_conv(h) # 通过26*1的卷积后，y的形状：[batch_size, 1, 1, 23]

        y = y.squeeze(1) # 去掉多余的维度，y形状：[batch_size, 1, 23]

        # 移除对时间步的嵌入编码操作
        # iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        # step_embedding = self.step_embedding(iter_step_tensor) # 已移除

        # 拼接时仅使用 x 与 y
        # 原为 concat_op((x, step_embedding, y))，去掉 step_embedding 后为 concat_op((x, y))
        concat_op = mindspore.ops.Concat(axis=1)
        bias_input = concat_op((x, y)) # bias_input的形状：[batch_size, 31, 23] (原为32, 现在少了一步)
        
        bias_input = bias_input.unsqueeze(1) # [batch_size, 1, 31, 23]
        bias_input = self.tcn(bias_input)     # 经过tcn后形状可能为：[batch_size, 1, reduced_time, 23]

        bias_input = bias_input.squeeze(1)    # [batch_size, reduced_time, 23]
        
        bias_output = self.bias_block(bias_input.unsqueeze(1)) # [batch_size, 1, 1, 23]
        bias_output = bias_output.squeeze(1) # [batch_size, 1, 23]

        # 加上偏差块的预测结果
        y = y + bias_output # [batch_size, 1, 23]

        return y # 返回计算结果
# 定义模型类
class TCN_MLP_Without_Spatial_MLP(nn.Cell):
    def __init__(self): # 构造方法
        super().__init__()
        # 偏差块与时间步嵌入保持不变
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(28,1), pad_mode='valid')
        )
        # 对预测时间步数据的嵌入编码操作保持不变
        self.step_embedding = nn.Embedding(horizon, sensor_num)

        # 移除空间 MLP 层，不再对 x 进行传感器间关联特征提取
        # self.spatial_mlp = nn.SequentialCell(
        #     nn.Dense(sensor_num, 128),
        #     nn.ReLU(),
        #     nn.Dense(128, 64),
        #     nn.ReLU(),
        #     nn.Dense(64, 32),
        #     nn.ReLU(),
        #     nn.Dense(32, sensor_num)
        # )

        # 时间卷积网络保持不变
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        )

        # 最终卷积保持不变
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid')

    def construct(self, x, iter_step):
        # 不再调用 spatial_mlp(x)，直接使用 x
        # 原逻辑: h = self.spatial_mlp(x)
        # h = x + h
        # 移除上述两行，直接使用 h = x
        h = x  

        h = h.unsqueeze(1)  # [batch_size, 1, 30, 23]
        h = self.tcn(h)      # [batch_size, 1, 26, 23]
        y = self.final_conv(h) # [batch_size, 1, 1, 23]
        y = y.squeeze(1)     # [batch_size, 1, 23]

        # 计算时间步数据的嵌入编码
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor) # [batch_size,1,23]

        concat_op = mindspore.ops.Concat(axis=1)
        # 拼接时仍使用 x, step_embedding, y
        # 这里 x 的形状为 [batch_size, 30, 23]
        # step_embedding 的形状 [batch_size, 1, 23]
        # y 的形状 [batch_size, 1, 23]
        # 拼接结果 [batch_size, 32, 23]
        bias_input = concat_op((x, step_embedding, y))
        bias_input = bias_input.unsqueeze(1) # [batch_size, 1, 32, 23]
        bias_input = self.tcn(bias_input)     # 经过tcn处理
        bias_input = bias_input.squeeze(1)    # 回到 [batch_size, reduced_time, 23]

        bias_output = self.bias_block(bias_input.unsqueeze(1)) # [batch_size, 1, 1, 23]
        bias_output = bias_output.squeeze(1) # [batch_size, 1, 23]

        y = y + bias_output # [batch_size, 1, 23]

        return y
class TCN_MLP_Without_TCN(nn.Cell):
    
    def __init__(self): #构造方法
        super().__init__() #调用父类的构造方法
        # 偏差块保持不变，但修改最后卷积核大小为(32,1)，因为不再有tcn降维
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32,1), pad_mode='valid')
        )
        # 对预测时间步数据的嵌入编码操作保持不变
        self.step_embedding = nn.Embedding(horizon, sensor_num)

        # 保留空间 MLP 层（spatial_mlp）
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )

        # 移除时序卷积层 tcn
        # self.tcn = nn.SequentialCell(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        # )

        # final_conv使用(30,1)卷积核代替原本的(26,1)，直接将30步压缩到1步
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(30, 1), pad_mode='valid')

    def construct(self, x, iter_step): #construct方法
        h = self.spatial_mlp(x) # 经过spatial_mlp提取空间特征 [batch_size,30,23]
        h = x + h  # 残差连接 [batch_size,30,23]

        h = h.unsqueeze(1)  # 升维为4D [batch_size,1,30,23]
        # 去掉对 h 的 tcn 调用，直接使用 h 进入 final_conv
        y = self.final_conv(h) # 使用30x1卷积核将30步降到1步 [batch_size,1,1,23]
        y = y.squeeze(1)       # [batch_size,1,23]

        # 计算时间步数据的嵌入编码
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor) # [batch_size,1,23]

        concat_op = mindspore.ops.Concat(axis=1)
        bias_input = concat_op((x, step_embedding, y)) # [batch_size,32,23]
        bias_input = bias_input.unsqueeze(1)           # [batch_size,1,32,23]

        # 去掉对 bias_input 的 tcn 调用，直接进入 bias_block
        bias_output = self.bias_block(bias_input)      # 卷积核(32,1)，输出 [batch_size,1,1,23]
        bias_output = bias_output.squeeze(1)           # [batch_size,1,23]

        # 加上偏差块的预测结果
        y = y + bias_output # [batch_size,1,23]

        return y
class TCN_MLP_Without_Final_Conv(nn.Cell): #定义TCN_MLP_with_Bias_Block类
    def __init__(self): #构造方法
        super().__init__() #调用父类的构造方法
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(28,1), pad_mode='valid')
        )
        self.step_embedding = nn.Embedding(horizon, sensor_num)
        
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )

        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        )

        # 移除 final_conv，不再使用最终卷积层
        # self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid') 

    def construct(self, x, iter_step):
        # 经过空间MLP和残差连接
        h = self.spatial_mlp(x) # [batch_size,30,23]
        h = x + h                # [batch_size,30,23]

        # 升维后通过tcn提取时间特征
        h = h.unsqueeze(1)       # [batch_size,1,30,23]
        h = self.tcn(h)          # [batch_size,1,26,23]

        # 不再通过final_conv进行降维，这里从最后一个时间步提取特征
        # h[:, :, -1, :]会选取最后一个时间步的特征，形状为[batch_size,1,23]
        y = h[:, :, -1, :]  # [batch_size,1,23]

        # 计算时间步的嵌入编码
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor) # [batch_size,1,23]

        concat_op = mindspore.ops.Concat(axis=1)
        # 拼接 x, step_embedding, y
        # x:[batch_size,30,23], step_embedding:[batch_size,1,23], y:[batch_size,1,23]
        # 拼接后为[batch_size,32,23]
        bias_input = concat_op((x, step_embedding, y))
        bias_input = bias_input.unsqueeze(1) # [batch_size,1,32,23]

        # 通过tcn后再进入bias_block
        bias_input = self.tcn(bias_input)     # [batch_size,1,26,23]
        bias_input = bias_input.squeeze(1)    # [batch_size,26,23]

        bias_output = self.bias_block(bias_input.unsqueeze(1)) # [batch_size,1,1,23]
        bias_output = bias_output.squeeze(1) # [batch_size,1,23]

        # 最终输出 y + bias_output
        y = y + bias_output # [batch_size,1,23]
        return y
class TCN_MLP_Basic(nn.Cell): #定义TCN_MLP_with_Bias_Block类
    def __init__(self):
        super().__init__()
        sensor_num = 23   # 假设原先定义在外部
        horizon = 5       # 假设原先定义在外部

        # 原先的bias_block组件被移除，不再定义bias_block
        # self.bias_block = nn.SequentialCell(...)

        # 移除step_embedding，不再对时间步进行嵌入编码
        # self.step_embedding = nn.Embedding(horizon, sensor_num)

        # 空间MLP保留
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )
        # 时间卷积层保留
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), pad_mode='valid'),
        )
        # 最终卷积层保留
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid')

    def construct(self, x, iter_step):
        # 空间特征提取并残差连接
        h = self.spatial_mlp(x) # [batch_size,30,23]
        h = x + h               # [batch_size,30,23]

        # 升维后进行时序卷积特征提取
        h = h.unsqueeze(1)      # [batch_size,1,30,23]
        h = self.tcn(h)         # [batch_size,1,26,23]

        # 使用final_conv得到最终预测结果
        y = self.final_conv(h)  # [batch_size,1,1,23]
        y = y.squeeze(1)        # [batch_size,1,23]

        # 移除 step_embedding，不计算iter_step_tensor和step_embedding
        # 移除 bias_block，不进行拼接 x、step_embedding、y ，不进行第二次 tcn 和不经过bias_block修正

        # 因为已经没有bias_input和bias_block的修正过程，此时 y 就是最终输出
        return y
        
    def construct(self, x):  # construct 方法
        h = self.spatial_mlp(x)  # 经过 spatial_mlp 处理，得到 h，形状：[batch_size, 30, sensor_num]
        h = x + h  # 残差连接，形状保持不变
        h = h.unsqueeze(1)  # 扩展维度，形状变为：[batch_size, 1, 30, sensor_num]
        h = self.tcn(h)  # 经过时序卷积层，形状：[batch_size, 1, 26, sensor_num]
        y = self.final_conv(h)  # 最终卷积层，形状：[batch_size, 1, 1, sensor_num]
        y = y.squeeze(1)  # 去掉第一个维度，形状：[batch_size, 1, sensor_num]
        return y  # 返回预测结果 y，形状：[batch_size, 1, sensor_num]
# 设置MindSpore运行模式
mindspore.set_context(mode=mindspore.GRAPH_MODE)

# 定义模型运行类，加入动态学习率和早停机制
class MULTI_STEP_MODEL_RUN:
    def __init__(self, model, loss_fn, optimizer=None, grad_fn=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = grad_fn

    def _train_one_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss.asnumpy()

    def _train_one_epoch(self, train_dataset):
        self.model.set_train(True)
        epoch_loss = 0
        for data, label in train_dataset.create_tuple_iterator():
            loss = self._train_one_step(data, label)
            epoch_loss += loss
        return epoch_loss / train_dataset.get_dataset_size()

    def evaluate(self, dataset):
        self.model.set_train(False)
        ls_pred, ls_label = [], []
        for data, label in dataset.create_tuple_iterator():
            muti_step_pred = mindspore.numpy.zeros_like(label[:, :, PV_index + DV_index])
            x = data
            for step in range(horizon):
                pred = self.model(x)
                muti_step_pred[:, step:step + 1, :] = pred[:, :, PV_index + DV_index]
                concat_op = mindspore.ops.Concat(axis=1)
                x = concat_op((x[:, 1:, :], pred))
                x[:, -1:, OP_index] = label[:, step:step + 1, OP_index]
            ls_pred += list(muti_step_pred.asnumpy())
            ls_label += list(label[:, :, PV_index + DV_index].asnumpy())
        eval_loss = self.loss_fn(Tensor(ls_pred), Tensor(ls_label)).asnumpy()
        return eval_loss, np.array(ls_pred), np.array(ls_label)

    def train(self, train_dataset, val_dataset, max_epoch_num, ckpt_file_path, patience=5):
        min_loss = float('inf')
        no_improve_count = 0  # 记录验证集上损失未改善的次数

        # 动态学习率调整器
        lr_scheduler = mindspore.nn.exponential_decay_lr(
            learning_rate=1e-3, decay_rate=0.9, total_step=max_epoch_num, step_per_epoch=1, decay_epoch=10
        )
        print('开始训练......')
        for epoch in range(1, max_epoch_num + 1):
            # 更新学习率
            lr = lr_scheduler[epoch - 1]
            
            self.optimizer.learning_rate = Tensor(lr, mindspore.float32)
            print(f'开始第 {epoch}/{max_epoch_num} 轮训练，学习率：{lr}')
            start_time = time.time()
            train_loss = self._train_one_epoch(train_dataset)
            eval_loss, _, _ = self.evaluate(val_dataset)
            print(f'训练集损失：{train_loss:.6f}，验证集损失：{eval_loss:.6f}')

            # 早停机制
            if eval_loss < min_loss:
                mindspore.save_checkpoint(self.model, ckpt_file_path)
                min_loss = eval_loss
                no_improve_count = 0
                print(f'验证集损失降低，保存模型到 {ckpt_file_path}')
            else:
                no_improve_count += 1
                print(f'验证集损失未降低，连续 {no_improve_count} 次未提升')
                if no_improve_count >= patience:
                    print('验证集损失连续多次未提升，提前停止训练')
                    break

            epoch_time = time.time() - start_time
            print(f'第 {epoch} 轮训练完成，耗时 {epoch_time:.2f} 秒')
        print('训练完成！')

    def test(self, test_dataset, ckpt_file_path):
        mindspore.load_checkpoint(ckpt_file_path, net=self.model)
        loss, preds, labels = self.evaluate(test_dataset)
        return loss, preds, labels

# 创建模型、损失函数和优化器
tcn_mlp_bias_more = TCN_MLP_Basic()
loss_fn = nn.MAELoss()

# 设置优化器，这里不再设置学习率，因为我们将使用学习率调度器
multi_step_optimizer = nn.Adam(tcn_mlp_bias_more.trainable_params(), learning_rate=1e-3)

# 定义前向函数和梯度函数
def multi_step_forward_fn(data, label):
    muti_step_pred = mindspore.numpy.zeros_like(label[:, :, PV_index + DV_index])
    x = data
    for step in range(horizon):
        pred = tcn_mlp_bias_more(x)
        muti_step_pred[:, step:step + 1, :] = pred[:, :, PV_index + DV_index]
        concat_op = mindspore.ops.Concat(axis=1)
        x = concat_op((x[:, 1:, :], pred))
        x[:, -1:, OP_index] = label[:, step:step + 1, OP_index]
    loss = loss_fn(muti_step_pred, label[:, :, PV_index + DV_index])
    return loss, muti_step_pred

multi_step_grad_fn = mindspore.value_and_grad(multi_step_forward_fn, None, multi_step_optimizer.parameters, has_aux=True)

# 创建模型运行实例
multi_step_model_run = MULTI_STEP_MODEL_RUN(tcn_mlp_bias_more, loss_fn, multi_step_optimizer, multi_step_grad_fn)

# 训练模型，设置早停参数patience为3
multi_step_model_run.train(train_dataset_t2, val_dataset_t2, max_epoch_num=50, ckpt_file_path='tcn_mlp_bias_More.ckpt', patience=5)

# 测试模型
tcn_mlp_bias = TCN_MLP_Basic()
multi_step_model_run = MULTI_STEP_MODEL_RUN(tcn_mlp_bias, loss_fn)

train_loss, _, _ = multi_step_model_run.test(train_dataset_t2, 'tcn_mlp_bias_More.ckpt')
val_loss, _, _ = multi_step_model_run.test(val_dataset_t2, 'tcn_mlp_bias_More.ckpt')
test_loss, preds, labels = multi_step_model_run.test(test_dataset_t2, 'tcn_mlp_bias_More.ckpt')
print(f'训练集损失：{train_loss}，验证集损失：{val_loss}，测试集损失：{test_loss}')

# 绘制预测结果图像
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
_, axes = plt.subplots(5, 1, figsize=(8, 16))
interval = int(horizon / 5)
for step in range(5):
    axes[step].set_title(f'第{step * interval + 1}个时间步的预测结果')
    axes[step].plot(range(1, 101), preds[:100, step * interval, 0], color='Red', label='预测值')
    axes[step].plot(range(1, 101), labels[:100, step * interval, 0], color='Blue', label='实际值')
    axes[step].legend()
plt.tight_layout()
plt.show()
