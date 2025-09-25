import numpy as np
import mindspore
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from mindspore import nn
import matplotlib.pyplot as plt
import time

# 设置为图模式并使用CPU，以加快在无GPU环境下的执行速度
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 设置随机种子以保证实验可复现
np.random.seed(0)
mindspore.set_seed(0)

sensor_num = 23  # 传感器数量
horizon = 5  # 预测的时间步数
PV_index = [idx for idx in range(9)]  # PV 变量的索引值范围
OP_index = [idx for idx in range(9, 18)]  # OP 变量的索引值范围
DV_index = [idx for idx in range(18, sensor_num)]  # DV 变量的索引值范围
data_path = 'D:\\VSC_code\\train.csv'  # 数据文件路径

# 读取数据（忽略第1行的标题及第1列的时间戳）
data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=range(1, sensor_num+1))
print('数据形状：{0}，元素类型：{1}'.format(data.shape, data.dtype))

def generateData(data, X_len, Y_len, sensor_num):
    point_num = data.shape[0]
    sample_num = point_num - X_len - Y_len + 1
    X = np.zeros((sample_num, X_len, sensor_num))
    Y = np.zeros((sample_num, Y_len, sensor_num))
    for i in range(sample_num):
        X[i] = data[i:i+X_len]
        Y[i] = data[i+X_len:i+X_len+Y_len]
    return X, Y

X_t2, Y_t2 = generateData(data, 30, horizon, sensor_num)
print('任务数据集输入数据形状：{0}，输出数据形状：{1}'.format(X_t2.shape, Y_t2.shape))

def splitData(X, Y):
    N = X.shape[0]
    train_X, train_Y = X[:int(N*0.6)], Y[:int(N*0.6)]
    val_X, val_Y = X[int(N*0.6):int(N*0.8)], Y[int(N*0.6):int(N*0.8)]
    test_X, test_Y = X[int(N*0.8):], Y[int(N*0.8):]
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

train_X_t2, train_Y_t2, val_X_t2, val_Y_t2, test_X_t2, test_Y_t2 = splitData(X_t2, Y_t2)
s = '训练集样本数：{0}，验证集样本数：{1}，测试集样本数：{2}'
print(s.format(train_X_t2.shape[0], val_X_t2.shape[0], test_X_t2.shape[0]))

class MultiTimeSeriesDataset():
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def generateMindsporeDataset(X, Y, batch_size):
    dataset = MultiTimeSeriesDataset(X.astype(np.float32), Y.astype(np.float32))
    # 使用并行数据加载和预取以加快数据读取
    dataset = GeneratorDataset(dataset, column_names=['data', 'label'], num_parallel_workers=4)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False, num_parallel_workers=4)
    dataset = dataset.prefetch(2)  # 预取数据
    return dataset

# 增大 batch_size，加快训练迭代
batch_size = 64

train_dataset_t2 = generateMindsporeDataset(train_X_t2, train_Y_t2, batch_size=batch_size)
val_dataset_t2 = generateMindsporeDataset(val_X_t2, val_Y_t2, batch_size=batch_size)
test_dataset_t2 = generateMindsporeDataset(test_X_t2, test_Y_t2, batch_size=batch_size)

for data, label in train_dataset_t2.create_tuple_iterator():
    print("对于任务二：")
    print('数据形状：', data.shape, '，数据类型：', data.dtype)
    print('标签形状：', label.shape, '，数据类型：', label.dtype)
    break

class TCN_MLP_with_Bias_Block_More(nn.Cell):
    def __init__(self):
        super().__init__()
        # Bias Block
        self.bias_block = nn.SequentialCell(
            nn.Dense(sensor_num, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(28, 1), pad_mode='valid')
        )
        # Step Embedding
        self.step_embedding = nn.Embedding(horizon, sensor_num)
        # Spatial MLP
        self.spatial_mlp = nn.SequentialCell(
            nn.Dense(sensor_num, 128),
            nn.ReLU(),
            nn.Dense(128, 64),
            nn.ReLU(),
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, sensor_num)
        )
        # TCN
        self.tcn = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), pad_mode='valid'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), pad_mode='valid'),
        )
        # Final Conv
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(26, 1), pad_mode='valid')

    def construct(self, x, iter_step):
        h = self.spatial_mlp(x)
        h = x + h  # 残差连接
        h = h.unsqueeze(1)  # [batch_size, 1, 30, 23]
        h = self.tcn(h)  # [batch_size, 1, 26, 23]
        y = self.final_conv(h)  # [batch_size, 1, 1, 23]
        y = y.squeeze(1)  # [batch_size, 1, 23]

        # Step Embedding
        iter_step_tensor = mindspore.numpy.full((x.shape[0], 1), iter_step, dtype=mindspore.int32)
        step_embedding = self.step_embedding(iter_step_tensor)  # [batch_size, 1, 23]

        concat_op = mindspore.ops.Concat(axis=1)
        bias_input = concat_op((x, step_embedding, y))  # [batch_size, 32, 23]
        bias_input = bias_input.unsqueeze(1)  # [batch_size, 1, 32, 23]
        bias_input = self.tcn(bias_input)  # [batch_size, 1, 26, 23]
        bias_input = bias_input.squeeze(1)  # [batch_size, 26, 23]

        bias_output = self.bias_block(bias_input.unsqueeze(1))  # [batch_size, 1, 1, 23]
        bias_output = bias_output.squeeze(1)  # [batch_size, 1, 23]

        y = y + bias_output
        return y

class MULTI_STEP_MODEL_RUN:
    def __init__(self, model, loss_fn, optimizer=None, grad_fn=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = grad_fn

    def _train_one_step(self, data, label):
        (loss, _), grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss

    def _train_one_epoch(self, train_dataset):
        self.model.set_train(True)
        for data, label in train_dataset.create_tuple_iterator():
            self._train_one_step(data, label)

    def evaluate(self, dataset):
        self.model.set_train(False)
        ls_pred, ls_label = [], []
        for data, label in dataset.create_tuple_iterator():
            muti_step_pred = mindspore.numpy.zeros_like(label[:, :, PV_index + DV_index])
            x = data
            for step in range(horizon):
                pred = self.model(x, step)
                muti_step_pred[:, step:step+1, :] = pred[:, :, PV_index + DV_index]
                concat_op = mindspore.ops.Concat(axis=1)
                x = concat_op((x[:, 1:, :], pred))
                x[:, -1:, OP_index] = label[:, step:step+1, OP_index]
            ls_pred += list(muti_step_pred.asnumpy())
            ls_label += list(label[:, :, PV_index + DV_index].asnumpy())
        return self.loss_fn(Tensor(ls_pred), Tensor(ls_label)), np.array(ls_pred), np.array(ls_label)

    def train(self, train_dataset, val_dataset, max_epoch_num, ckpt_file_path):
        min_loss = mindspore.Tensor(float('inf'), mindspore.float32)
        patience_counter = 0  # 早停计数器
        patience = 3  # 设置耐心值为 3
        print('开始训练......')
        for epoch in range(1, max_epoch_num + 1):
            print('第 {0}/{1} 轮'.format(epoch, max_epoch_num))
            start_time = time.time()
            self._train_one_epoch(train_dataset)
            train_loss, _, _ = self.evaluate(train_dataset)
            eval_loss, _, _ = self.evaluate(val_dataset)
            print('训练集损失：{0}，验证集损失：{1}'.format(train_loss.asnumpy(), eval_loss.asnumpy()))
            if eval_loss < min_loss:
                mindspore.save_checkpoint(self.model, ckpt_file_path)
                min_loss = eval_loss
                patience_counter = 0  # 重置耐心计数器
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('验证集损失在 {0} 轮没有降低，提前停止训练。'.format(patience))
                    break  # 早停
            epoch_time = time.time() - start_time
            print('第 {0} 轮训练完成，耗时 {1:.2f} 秒'.format(epoch, epoch_time))
        print('训练完成！')

    def test(self, test_dataset, ckpt_file_path):
        mindspore.load_checkpoint(ckpt_file_path, net=self.model)
        loss, preds, labels = self.evaluate(test_dataset)
        return loss, preds, labels

# 创建模型实例
tcn_mlp_bias_more = TCN_MLP_with_Bias_Block_More()

# 定义损失函数
loss_fn = nn.MAELoss()

# 定义优化器
multi_step_optimizer = nn.Adam(tcn_mlp_bias_more.trainable_params(), learning_rate=1e-3)

# 定义前向传播和梯度计算函数
def multi_step_forward_fn(data, label):
    muti_step_pred = mindspore.numpy.zeros_like(label[:, :, PV_index + DV_index])
    x = data
    for step in range(horizon):
        pred = tcn_mlp_bias_more(x, step)
        muti_step_pred[:, step:step+1, :] = pred[:, :, PV_index + DV_index]
        concat_op = mindspore.ops.Concat(axis=1)
        x = concat_op((x[:, 1:, :], pred))
        x[:, -1:, OP_index] = label[:, step:step+1, OP_index]
    loss = loss_fn(muti_step_pred, label[:, :, PV_index + DV_index])
    return loss, muti_step_pred

multi_step_grad_fn = mindspore.value_and_grad(multi_step_forward_fn, None, multi_step_optimizer.parameters, has_aux=True)

# 创建模型运行实例
multi_step_model_run = MULTI_STEP_MODEL_RUN(tcn_mlp_bias_more, loss_fn, multi_step_optimizer, multi_step_grad_fn)

# 开始训练，最大轮数为 10 (可以根据需要调整，减少轮数加快测试速度)
max_epochs = 10  
multi_step_model_run.train(train_dataset_t2, val_dataset_t2, max_epochs, 'tcn_mlp_bias_More.ckpt')

# 测试模型
train_loss, _, _ = multi_step_model_run.test(train_dataset_t2, 'tcn_mlp_bias_More.ckpt')
val_loss, _, _ = multi_step_model_run.test(val_dataset_t2, 'tcn_mlp_bias_More.ckpt')
test_loss, preds, labels = multi_step_model_run.test(test_dataset_t2, 'tcn_mlp_bias_More.ckpt')
print('训练集损失：{0}，验证集损失：{1}，测试集损失：{2}'.format(train_loss.asnumpy(), val_loss.asnumpy(), test_loss.asnumpy()))

# 绘制结果
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
_, axes = plt.subplots(5, 1, figsize=(8, 16))
interval = int(horizon / 5)
for step in range(5):
    axes[step].set_title('第%d个时间步的预测结果' % (step * interval + 1))
    axes[step].plot(range(1, 101), preds[:100, step * interval, 0], color='Red', label='预测值')
    axes[step].plot(range(1, 101), labels[:100, step * interval, 0], color='Blue', label='真实值')
    axes[step].legend()
plt.tight_layout()
plt.show()
