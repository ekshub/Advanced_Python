# mymatrix.py
import random

class Matrix:
    def __init__(self, data):
        self.data = data
        self.check_matrix()

    def check_matrix(self):
        """检查矩阵是否有效"""
        if self.data is None:
            raise ValueError("输入矩阵不能为空（None）")
        if not isinstance(self.data, list) or not all(isinstance(row, list) for row in self.data):
            raise ValueError("输入必须是一个二维列表")
        if len(self.data) == 0:
            raise ValueError("输入矩阵不能为空")
        if len(self.data[0]) == 0:
            raise ValueError("输入矩阵的第一行不能为空")

        num_cols = len(self.data[0])
        for row in self.data:
            if len(row) != num_cols:
                raise ValueError("输入矩阵的每一行必须具有相同的列数")

        for i, row in enumerate(self.data):
            for j, elem in enumerate(row):
                if not isinstance(elem, (int, float)):
                    raise ValueError(f"矩阵元素 ({i}, {j}) 不是数字: {elem}")

    def is_square(self):
        """判断矩阵是否为方阵"""
        return len(self.data) > 0 and all(len(row) == len(self.data) for row in self.data)
    
    def same_shape(self, other):
        """判断矩阵是否形状相同"""
        return len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0])


    @staticmethod
    def generate_zero_matrix(n, m):
        """创建一个n*m的零初始化矩阵"""
        return Matrix([[0 for _ in range(m)] for _ in range(n)])

    @staticmethod
    def generate_random_matrix(n, m, lower_bound=0, upper_bound=100):
        """创建一个n*m的随机初始化矩阵"""
        return Matrix([[random.randint(lower_bound, upper_bound) for _ in range(m)] for _ in range(n)])

    @staticmethod
    def generate_identity_matrix(n):
        """创建一个n*n的单位矩阵"""
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    def add(self, other):
        """计算矩阵相加"""
        if not self.same_shape(other):
            raise ValueError("矩阵相加要求两个矩阵形状相同")
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])

    def subtract(self, other):
        """计算矩阵相减"""
        if not self.same_shape(other):
            raise ValueError("矩阵相减要求两个矩阵形状相同")
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])

    def hadamard_product(self, other):
        """计算矩阵哈达玛积"""
        if not self.same_shape(other):
            raise ValueError("哈达玛积要求两个矩阵形状相同")
        return Matrix([[self.data[i][j] * other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])
    def multiply(self, other):
        """计算矩阵乘积"""
        if len(self.data[0]) != len(other.data):
            raise ValueError("矩阵乘法要求第一个矩阵的列数等于第二个矩阵的行数")

        result = [[0 for _ in range(len(other.data[0]))] for _ in range(len(self.data))]
        for i in range(len(self.data)):
            for j in range(len(other.data[0])):
                for k in range(len(other.data)):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(result)

    def transpose(self):
        """计算矩阵的转置"""
        return Matrix([[self.data[i][j] for i in range(len(self.data))] for j in range(len(self.data[0]))])
class SquareMatrix(Matrix):
    def __init__(self, data):
        super().__init__(data)  # 调用父类构造方法
        if not self.is_square():
            raise ValueError("矩阵必须为方阵")

    @staticmethod
    def generate_zero_square_matrix(n):
        """创建一个n*n的零初始化方阵"""
        return SquareMatrix([[0 for _ in range(n)] for _ in range(n)])
    @staticmethod
    def generate_random_matrix(n,  lower_bound=0, upper_bound=100):
        """创建一个n*n的随机初始化矩阵"""
        return SquareMatrix([[random.randint(lower_bound, upper_bound) for _ in range(n)] for _ in range(n)])

    @staticmethod
    def generate_identity_matrix(n):
        """创建一个n*n的单位矩阵"""
        return SquareMatrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    def determinant(self):
        """计算矩阵的行列式"""
        if not self.is_square():
            raise ValueError("矩阵必须为方阵")
        
        n = len(self.data)
        
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

        det = 0
        for c in range(n):
            minor = [[self.data[i][j] for j in range(n) if j != c] for i in range(1, n)]
            det += ((-1) ** c) * self.data[0][c] * SquareMatrix(minor).determinant()

        return det
    def gaussian_determinant(self):
        """计算矩阵的行列式"""
        n = len(self.data)

        if not self.is_square():
            raise ValueError("矩阵必须为方阵")

        A = [row[:] for row in self.data]
        det = 1

        for i in range(n):
            # 查找主元
            if A[i][i] == 0:
                for j in range(i + 1, n):
                    if A[j][i] != 0:
                        A[i], A[j] = A[j], A[i]  # 交换行
                        det *= -1  # 行交换会改变行列式的符号
                        break

            pivot = A[i][i]
            if pivot == 0:
                return 0

            for j in range(i + 1, n):
                ratio = A[j][i] / pivot
                for k in range(i, n):
                    A[j][k] -= ratio * A[i][k]

            det *= pivot

        return det

    def gaussian_inverse(self):
        """计算矩阵的逆"""
        if not self.is_square():
            raise ValueError("矩阵必须为方阵")

        n = len(self.data)  # 获取矩阵的行数（方阵的行数等于列数）

        # 创建增广矩阵，将原矩阵与单位矩阵拼接在一起
        aug_matrix = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(self.data)]

        # 执行高斯消元法以将增广矩阵转换为行最简形式
        for i in range(n):
            # 寻找主元
            max_row = i  # 初始化最大行索引
            for j in range(i + 1, n):
                # 找到绝对值最大的主元
                if abs(aug_matrix[j][i]) > abs(aug_matrix[max_row][i]):
                    max_row = j

            # 如果找到的主元行不等于当前行，进行交换
            if max_row != i:
                aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]  # 交换行

            # 检查主元是否为零
            if aug_matrix[i][i] == 0:
                raise ValueError("矩阵不可逆，主元为零")

            # 归一化当前主元行，使主元变为 1
            pivot = aug_matrix[i][i]
            for k in range(2 * n):
                aug_matrix[i][k] /= pivot  # 将当前行的每个元素除以主元的值

            # 消元，调整当前行以下的所有行
            for j in range(i + 1, n):
                factor = aug_matrix[j][i]  # 计算消元因子
                for k in range(2 * n):
                    aug_matrix[j][k] -= factor * aug_matrix[i][k]  # 消元，调整当前行

        # 进行反向消元，消去上三角中的元素
        for i in range(n - 1, -1, -1):  # 从最后一行开始向上处理
            for j in range(i - 1, -1, -1):  # 处理当前行以上的所有行
                factor = aug_matrix[j][i]  # 计算消元因子
                for k in range(2 * n):
                    aug_matrix[j][k] -= factor * aug_matrix[i][k]  # 消元，调整当前行

        # 提取逆矩阵，逆矩阵位于增广矩阵的右侧部分
        inverse_matrix = []
        for i in range(n):
            row = aug_matrix[i][n:]  # 取右半部分
            inverse_matrix.append(row)  # 将计算得到的行添加到逆矩阵中

        return inverse_matrix  # 返回计算得到的逆矩阵

    def get_adjugate(self):
        """计算伴随矩阵"""
        n = len(self.data)
        adjugate = [[0] * n for _ in range(n)]  # 创建零矩阵

        for i in range(n):
            for j in range(n):
                # 代数余子式
                minor = [r[:j] + r[j + 1:] for r in (self.data[:i] + self.data[i + 1:])]
                # 伴随矩阵的元素
                adjugate[j][i] = ((-1) ** (i + j)) * SquareMatrix(minor).gaussian_determinant()  # 计算余子式

        return adjugate

    def adjugate_inverse(self):
        """使用伴随矩阵法计算逆矩阵"""
        det = self.gaussian_determinant()

        if det == 0:
            raise ValueError("该矩阵不可逆。")

        adjugate = self.get_adjugate()
        inverse = [[adjugate[i][j] / det for j in range(len(adjugate))] for i in range(len(adjugate))]
        return inverse
    

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])
