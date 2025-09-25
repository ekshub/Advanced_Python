import numpy as np
import time
def generate_kernels(num_kernels, kernel_shape, num_channels):
    """随机生成卷积核"""
    return np.random.rand(num_kernels, num_channels, *kernel_shape).astype(np.float64)

def conv2d(input_data, kernels):
    """执行卷积运算"""
    num_channels, input_height, input_width = input_data.shape
    num_kernels, _, kernel_height, kernel_width = kernels.shape

    # 计算输出特征图的形状
    output_shape = (num_kernels, 
                    input_height - kernel_height + 1, 
                    input_width - kernel_width + 1)
    
    R = np.zeros(output_shape, dtype=np.float64)  # 初始化输出特征图为浮点数

    # 执行卷积运算
    for k in range(num_kernels):
        for row in range(output_shape[1]):
            for col in range(output_shape[2]):
                for c in range(num_channels):
                    R[k, row, col] += np.sum(
                        input_data[c, row:row + kernel_height, col:col + kernel_width] * kernels[k, c]
                    )

    return R
def test_conv2d():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 参数设置
    num_channels = 3    # 输入数据的通道数（例如 RGB 图像）
    input_height = 5    # 输入数据的高度
    input_width = 5     # 输入数据的宽度
    num_kernels = 2     # 卷积核数量
    kernel_shape = (2, 2)  # 卷积核的高度和宽度

    # 生成随机输入数据（形状: (num_channels, input_height, input_width)）
    input_data = np.random.rand(num_channels, input_height, input_width).astype(np.float64)

    # 生成随机卷积核（形状: (num_kernels, num_channels, kernel_height, kernel_width)）
    kernels = np.random.rand(num_kernels, num_channels, *kernel_shape).astype(np.float64)

    # 调用 conv2d 函数
    output = conv2d(input_data, kernels)

    # 输出结果
    print("输入数据：")
    print(input_data)
    print("\n卷积核：")
    print(kernels)
    print("\n输出特征图：")
    print(output)



def mse_loss(predicted, target):
    """均方误差损失"""
    return np.mean((predicted - target) ** 2)

def mse_loss_gradient(predicted, target):
    """均方误差损失的梯度"""
    return 2 * (predicted - target) / predicted.size

def conv2d_backprop(input_data, kernels, target_output, R):
    """反向传播，计算卷积核的梯度"""
    output_grad = mse_loss_gradient(R, target_output)
    num_kernels, num_channels, kernel_height, kernel_width = kernels.shape
    _, output_height, output_width = output_grad.shape

    # 初始化卷积核的梯度
    kernel_grad = np.zeros_like(kernels, dtype=np.float64)

    # 计算卷积核的梯度
    for k in range(num_kernels):
        for row in range(output_height):
            for col in range(output_width):
                for c in range(num_channels):
                    kernel_grad[k, c] += input_data[c, row:row + kernel_height, col:col + kernel_width] * output_grad[k, row, col]
    return kernel_grad

def load_data(input_file, target_file):
    """从文件中加载输入数据和目标输出"""
    input_data = np.load(input_file)  # shape: (num_samples, num_channels, height, width)
    target_output = np.load(target_file)  # shape: (num_samples, num_kernels, output_height, output_width)
    return input_data, target_output
import numpy as np

def test_load_data():
    # 1. 创建模拟数据
    num_samples = 5
    num_channels = 3
    height = 10
    width = 10
    num_kernels = 2
    output_height = height - 1  # 假设卷积核为 2x2
    output_width = width - 1

    # 模拟输入数据和目标输出
    input_data = np.random.rand(num_samples, num_channels, height, width)
    target_output = np.random.rand(num_samples, num_kernels, output_height, output_width)

    # 保存为 .npy 文件
    np.save('test_input_data.npy', input_data)
    np.save('test_target_output.npy', target_output)

    # 2. 调用 load_data 函数
    loaded_input, loaded_target = load_data('test_input_data.npy', 'test_target_output.npy')

    # 3. 验证加载的数据是否正确
    assert loaded_input.shape == input_data.shape, "输入数据形状不匹配"
    assert loaded_target.shape == target_output.shape, "目标输出形状不匹配"
    assert np.allclose(loaded_input, input_data), "输入数据内容不匹配"
    assert np.allclose(loaded_target, target_output), "目标输出内容不匹配"

    print("测试通过！数据加载正常。")

# 执行测试


def train_model(input_file, target_file, num_kernels, kernel_shape, learning_rate=0.1, num_epochs=100):
    """训练模型"""
    # 从文件中加载输入数据和目标输出
    input_data, target_output = load_data(input_file, target_file)

    # 随机生成卷积核
    kernels = generate_kernels(num_kernels, kernel_shape, input_data.shape[1])

    num_samples = input_data.shape[0]

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录开始时间
        epoch_loss = 0  # 记录每个轮次的总损失

        for i in range(num_samples):
            # 对每个样本进行卷积操作
            R = conv2d(input_data[i], kernels)

            # 计算损失
            loss = mse_loss(R, target_output[i])
            epoch_loss += loss

            # 反向传播，计算梯度
            kernel_grad = conv2d_backprop(input_data[i], kernels, target_output[i], R)

            # 更新卷积核
            kernels -= learning_rate * kernel_grad

        # 计算轮次的平均损失
        avg_loss = epoch_loss / num_samples

        # 计算并输出训练时间
        training_time = time.time() - start_time
        print(f'轮次: {epoch + 1}, 平均损失: {avg_loss:.4f}, 学习率: {learning_rate}, 训练时间: {training_time:.2f}秒')
    return kernels
if __name__=='__main__':
    train_model('input_data.npy', 'target_output.npy', num_kernels=2, kernel_shape=(2, 2), num_epochs=10)