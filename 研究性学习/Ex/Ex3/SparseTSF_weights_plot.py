import torch
import torch.nn as nn
import torchvision
import numpy as np
from plot_array import plot_weights


class SparseTSF(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, enc_in=1, period_len=24, model_type="linear"):
        super().__init__()

        # get parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len
        self.model_type = model_type
        assert self.model_type in ['linear']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)


    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y

def get_main_periods(
    data_batch,
    sample_rate=1.0,
    min_period=24,
    max_period=96,
    main_feature_index=0,
    main_feature_weight=2.0,
    penalty_exponent=1.2,  # 调整此参数来控制幂指数
    alpha=0.02,           # 可选的指数衰减参数，默认不使用指数衰减
    device='cuda'
):
    """
    该函数接受一个时间序列批次，计算所有特征的傅里叶变换，
    找到幅度最大的 5 个频率分量，并返回对应的周期（整数形式），
    优先选取更接近最小周期的值，并确保周期在指定的最小和最大范围内。

    参数：
    - data_batch: 输入数据，形状为 (batch_size, T, channels)
    - sample_rate: 采样率，默认为 1.0
    - min_period: 最小周期，默认为 24
    - max_period: 最大周期，默认为 96
    - main_feature_index: 主特征的索引（从 0 开始）
    - main_feature_weight: 主特征的权重，默认为 2.0
    - penalty_exponent: 惩罚的幂指数，默认为 1.2，可调整
    - alpha: 指数衰减参数，如果为 None，则不使用指数衰减，默认为 None
    - device: 设备，默认为 'cuda'
    """
    # 检查是否有可用的 GPU，如果没有则回退到 CPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("GPU 不可用，切换到 CPU。")
        device = 'cpu'
    
    # 确保数据在正确的设备上
    data_batch = data_batch.to(device)
    batch_size, T, channels = data_batch.shape

    # 对时间维度进行傅里叶变换
    fft_result = torch.fft.fft(data_batch, n=T, dim=1)  # 形状: (batch_size, T, channels)

    # 计算频率序列
    freqs = torch.fft.fftfreq(T, d=1 / sample_rate, device=device)  # 形状: (T,)
    positive_freqs = freqs[:T // 2]  # 只考虑正频率部分，形状: (T//2,)

    # 提取正频率部分的傅里叶变换结果
    fft_positive = fft_result[:, :T // 2, :]  # 形状: (batch_size, T//2, channels)

    # 计算幅度谱并对批次维度取平均，形状: (T//2, channels)
    magnitudes = torch.abs(fft_positive).mean(dim=0)

    # 增加主特征的权重
    magnitudes[:, main_feature_index] *= main_feature_weight

    # 聚合所有特征的幅度谱，形状: (T//2,)
    aggregated_magnitudes = magnitudes.mean(dim=1)

    # 计算对应的周期，并取整数，形状: (T//2,)
    periods = (1.0 / positive_freqs).round().int()

    # 处理可能的无穷大和 NaN 值（零频率会导致无穷大的周期）
    periods = torch.where(
        torch.isinf(periods) | torch.isnan(periods),
        torch.tensor(T, device=device, dtype=periods.dtype),
        periods
    )

    # 筛选周期在指定范围内的频率分量
    mask = (periods >= min_period) & (periods <= max_period)
    periods = periods[mask]
    aggregated_magnitudes = aggregated_magnitudes[mask]

    # 如果没有符合条件的频率分量，抛出异常或返回默认值
    if aggregated_magnitudes.numel() == 0:
        raise ValueError("在指定的周期范围内没有频率分量，请调整 min_period 和 max_period。")

    # 调整幅度以惩罚较长的周期，控制惩罚力度
    if alpha is not None:
        # 使用指数衰减函数来调整幅度
        adjusted_magnitudes = aggregated_magnitudes * torch.exp(-alpha * (periods.float() - min_period))
    else:
        # 使用幂指数来调整幅度，降低惩罚力度
        adjusted_magnitudes = aggregated_magnitudes / (periods.float() ** penalty_exponent)

    # 找到调整后幅度最大的 5 个频率分量
    topk = torch.topk(adjusted_magnitudes, k=5, largest=True)
    top_indices = topk.indices  # 形状: (5,)

    # 对应的周期
    top_periods = periods[top_indices]  # 形状: (5,)

    # 将周期按从小到大排序
    top_periods, _ = torch.sort(top_periods)

    # 返回形状为 (5,) 的张量，包含更接近最小周期的 5 个主要周期（整数）
    return top_periods
'''
def get_x(data_batch, low_freq_hz=0.1, high_freq_hz=None, sample_rate=1.0, device='cuda'):
    # 确保数据在正确的设备上
    data_batch = data_batch.to(device)
    batch_num, T, channels = data_batch.shape

    # 计算频率分辨率和频率序列
    freqs = np.fft.fftfreq(T, d=1 / sample_rate)

    # 确定高频阈值
    if high_freq_hz is None or high_freq_hz > sample_rate / 2:
        high_freq_hz = sample_rate / 2

    # 只取正频率部分
    positive_freqs = freqs[:T // 2]
    freq_indices = np.where((positive_freqs >= low_freq_hz) & (positive_freqs <= high_freq_hz))[0]

    # 确保至少保留一个频率
    if freq_indices.size == 0:
        raise ValueError("指定的频率范围内没有频率分量。请调整 low_freq_hz 和 high_freq_hz。")

    # 对时间维度进行傅里叶变换
    fft_result = torch.fft.fft(data_batch, dim=1)
    fft_result = fft_result[:, freq_indices, :]

    return fft_result


class PeriodExtractor(nn.Module):
    def __init__(self, k=5, min_period=24, max_period=100):
        super(PeriodExtractor, self).__init__()
        self.k = k
        self.min_period = min_period
        self.max_period = max_period

        # Convolutional layers to process the magnitude spectrum
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Adaptive pooling to reduce the frequency dimension
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=32)

        # Fully connected layers to map features to period outputs
        self.fc1 = nn.Linear(64 * 32, 128)
        self.fc2 = nn.Linear(128, self.k)

    def forward(self, fft_data):
        """
        fft_data: shape (batch_size, s, channels)
        """
        # Compute magnitude spectrum
        magnitude = torch.abs(fft_data)  # (batch_size, s, channels)

        # Average over channels
        avg_magnitude = magnitude.mean(dim=2)  # (batch_size, s)

        # Add a channel dimension for Conv1d input
        x = avg_magnitude.unsqueeze(1)  # (batch_size, 1, s)
        if x.dtype != torch.float32:
            x = x.float()

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))  # (batch_size, 32, s)
        x = F.relu(self.conv2(x))  # (batch_size, 64, s)

        # Adaptive pooling
        x = self.global_pool(x)  # (batch_size, 64, 32)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 64 * 32)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, k)

        # Ensure periods are within the specified range
        periods = torch.round(torch.relu(x)).int()  # (batch_size, k)
        periods = torch.clamp(periods, min=self.min_period, max=self.max_period)

        # Average over the batch, output shape (1, k)
        output_periods = periods.float().mean(dim=0, keepdim=True)

        return output_periods
'''

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 获取参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.data = configs.data

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
       
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, 
            padding=self.period_len // 2, 
            padding_mode="zeros", 
            bias=False
        )
        self.conv1d2 = nn.Conv1d(
            in_channels=5, 
            out_channels=1, 
            kernel_size=5, 
            stride=1, 
            padding=2
        )
        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
           
        # 获取主要周期
        main_periods = get_main_periods(
            x, 
            min_period=self.period_len, 
            main_feature_index=self.enc_in - 1
        )  

        # 将 main_periods 保存到当前文件夹的文件中
        # 根据 self.data 和 self.pred_len 生成文件名
        filename = f"{self.data}_pred{self.period_len}.csv"
        filepath = os.path.join(os.getcwd(), filename)

        # 将 main_periods 转换为整数列表
        main_periods_list = [int(p.item()) for p in main_periods]

        # 将数据以逗号分隔的形式写入文件，添加到文件末尾
        with open(filepath, 'a') as f:
            f.write(','.join(map(str, main_periods_list)) + '\n')

        # 归一化和维度变换
        seq_mean = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, channels)
        x = (x - seq_mean).permute(0, 2, 1)  # (batch_size, channels, seq_len)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        
        z = []  # 用于存储计算得到的张量 y

        for i in range(5):
            # 提取主要周期的标量值
            period = int(main_periods[i].item())
           
            # 计算分段数量
            seg_num_x1 = self.seq_len // period
      
            # 计算余数并确保为整数
            remainder = int(self.seq_len % period)

            if remainder != 0:
                x_i = x[:, :, :-remainder]  # 截断多余的元素
            else:
                x_i = x

            x_i = x_i.reshape(-1, seg_num_x1, period).permute(0, 2, 1)
            x_i = x_i.to(torch.float32)

            # 截断第二维度
            if x_i.shape[1] > self.period_len:
                x_i = x_i[:, :self.period_len, :]

            # 填充第三维度
            if x_i.shape[2] < self.seg_num_x:
                padding_size = self.seg_num_x - x_i.shape[2]
                x_i = torch.nn.functional.pad(x_i, (0, padding_size))

            x_i = x_i.reshape(batch_size * self.enc_in, self.period_len, self.seg_num_x)
            # 稀疏预测
            y = self.linear(x_i)  # (batch_size * channels, period_len, m)

            # 上采样并恢复形状
            y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

            # 反归一化
            y = y.permute(0, 2, 1) + seq_mean  # (batch_size, self.pred_len, channels)
            z.append(y)

        # 将结果堆叠并调整维度
        z = torch.stack(z)  # (5, batch_size, self.pred_len, channels)
        z = z.permute(1, 3, 2, 0)  # (batch_size, channels, self.pred_len, 5)

        # 合并维度并应用卷积层
        z = z.contiguous().view(batch_size, self.enc_in * self.pred_len, 5)
        z = z.permute(0, 2, 1)  # (batch_size, 5, enc_in * pred_len)
        z = z.to(torch.float32)
        
        output = self.conv1d2(z)

        # 恢复输出形状
        output = output.squeeze(1)  # (batch_size, enc_in * pred_len)
        
        output = output.view(batch_size, self.enc_in, self.pred_len)  # (batch_size, channels, pred_len)
        output = output.permute(0, 2, 1)
        return output
if __name__ == "__main__":
    config=Config()
    net = Model(config)
    
    net.load_state_dict(torch.load('/home/ma-user/work/SparseTSF/checkpoints/ETTh1_96_96_MyModel_ETTh1_ftM_sl96_pl96_linear_test_0_seed2023/checkpoint.pth', map_location='cpu'))
    weights = np.zeros((96, 96))

    for i in range(96):
        x = np.zeros((1, 96, 1), dtype=np.float32)  
        x[0, i, 0] = 1  
        x_tensor = torch.tensor(x)
        output = net(x_tensor)
        weights[i] = output.detach().numpy().flatten()  
        weights_matrix = np.array(weights)
    
    plot_weights(weights_matrix,'111weights_plot_no_norm_xxxx111.png',norm=True)
    
    

