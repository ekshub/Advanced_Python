import numpy as np

def generate_random_data(input_shape, target_shape, input_file='./input_data.npy', target_file='./target_output.npy'):
    """
    随机生成输入数据和对应的预期输出，并存储为npy文件。

    """
    # 随机生成输入数据
    input_data = np.random.rand(*input_shape).astype(np.float64)

    # 随机生成目标输出数据
    target_output = np.random.rand(*target_shape).astype(np.float64)

    # 保存数据到npy文件
    np.save(input_file, input_data)
    np.save(target_file, target_output)

    print(f'输入数据已保存到 {input_file}')
    print(f'目标输出数据已保存到 {target_file}')

input_shape = (100, 3, 28, 28)  # 10个样本，3个通道，28x28的图像
target_shape = (100, 2, 27, 27)  # 10个样本，2个卷积核，27x27的输出特征图
generate_random_data(input_shape, target_shape)
