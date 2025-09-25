# main.py
import mymatrix as mm
import random
import numpy as np
import time

def test_matrix_functions(num_tests=100):
    """测试所有矩阵函数的准确率"""
    
    # 测试 matrix_multiply
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 100)
        size2=random.randint(1, 100) 
        size3=random.randint(1, 100)
        A = mm.Matrix.generate_random_matrix(size1, size2)
        B = mm.Matrix.generate_random_matrix(size2, size3)
        try:
            custom_result = A.multiply(B)
            numpy_result = np.matmul(A, B)
            if np.allclose(custom_result, numpy_result.tolist()):
                correct_count += 1
        except ValueError:
            continue  # 如果矩阵不符合乘法条件，跳过

    print(f"Matrix Multiply Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 add_matrices
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 100)
        size2=random.randint(1, 100)
        A = mm.generate_random_matrix(size1, size2)
        B = mm.generate_random_matrix(size1, size2)
        try:
            custom_result = mm.add_matrices(A, B)
            numpy_result = np.add(A, B).tolist()
            if np.allclose(custom_result, numpy_result):
                correct_count += 1
        except ValueError:
            continue  # 跳过不符合条件的测试

    print(f"Add Matrices Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 subtract_matrices
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 100)
        size2=random.randint(1, 100)
        A = mm.generate_random_matrix(size1, size2)
        B = mm.generate_random_matrix(size1, size2)
        try:
            custom_result = mm.subtract_matrices(A, B)
            numpy_result = np.subtract(A, B).tolist()
            if np.allclose(custom_result, numpy_result):
                correct_count += 1
        except ValueError:
            continue

    print(f"Subtract Matrices Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 hadamard_product
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 100)
        size2=random.randint(1, 100)
        A = mm.generate_random_matrix(size1, size2)
        B = mm.generate_random_matrix(size1, size2)
        try:
            custom_result = mm.hadamard_product(A, B)
            numpy_result = np.multiply(A, B).tolist()
            if np.allclose(custom_result, numpy_result):
                correct_count += 1
        except ValueError:
            continue

    print(f"Hadamard Product Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 transpose
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 100)
        size2=random.randint(1, 100)
        A = mm.generate_random_matrix(size1, size2)
        
        custom_result = mm.transpose(A)
        numpy_result = np.transpose(A).tolist()
        if np.allclose(custom_result, numpy_result):
            correct_count += 1

    print(f"Transpose Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 determinant
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 5)
        A = mm.generate_random_matrix(size1, size1)
        try:
            custom_result = mm.determinant(A)
            numpy_result = np.linalg.det(A)
            if np.isclose(custom_result, numpy_result):
                correct_count += 1
        except ValueError:
            continue
    print(f"Determinant Accuracy: {correct_count / num_tests * 100:.2f}%")
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 5)
        A = mm.generate_random_matrix(size1, size1)
        try:
            custom_result = mm.gaussian_determinant(A)
            numpy_result = np.linalg.det(A)
            if np.isclose(custom_result, numpy_result):
                correct_count += 1
        except ValueError:
            continue
    print(f"Gaussian Determinant Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 matrix_inverse
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 100)
        A = mm.generate_random_matrix(size1, size1)
        try:
            # 确保矩阵可逆
                custom_result = mm.gaussian_matrix_inverse(A)
                numpy_result = np.linalg.inv(A).tolist()
                if np.allclose(custom_result, numpy_result):
                    correct_count += 1
        except ValueError:
            continue

    print(f"Gaussian Matrix Inverse Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 get_adjugate
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 5)
        A = mm.generate_random_matrix(size1, size1)
        try:
            custom_result = mm.get_adjugate(A)
            numpy_result = np.linalg.inv(A) * mm.determinant(A)  # adjugate(A) = inv(A) * det(A)
            if np.allclose(custom_result, numpy_result.tolist()):
                correct_count += 1
        except ValueError:
            continue

    print(f"Adjugate Accuracy: {correct_count / num_tests * 100:.2f}%")

    # 测试 matrix_inverse2
    correct_count = 0
    for _ in range(num_tests):
        size1=random.randint(1, 5)
        A = mm.generate_random_matrix(size1, size1)
        try:
            if mm.determinant(A) != 0:
                custom_result = mm.adjugate_matrix_inverse(A)
                numpy_result = np.linalg.inv(A).tolist()
                if np.allclose(custom_result, numpy_result):
                    correct_count += 1
        except ValueError:
            continue

    print(f"Adjugate Matrix Inverse Accuracy: {correct_count / num_tests * 100:.2f}%") 
def time_function(func, *args):
    """计算函数执行时间"""
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return (end_time - start_time)  # 返回每次调用的平均时间

# 测试每个函数的时间开销
def test_performance():
    """测试各个矩阵操作函数的性能"""

    matrix = mm.SquareMatrix.generate_random_matrix(8)
    print(f"Testing performance with a {8}x{8} matrix:")
    print(f"Determinant: {time_function(matrix.determinant):.6f} seconds")
    print("-" * 40)

    matrix = mm.SquareMatrix.generate_random_matrix(10)
    print(f"Testing performance with a {10}x{10} matrix:")
    print(f"Adjugate: {time_function(matrix.get_adjugate):.6f} seconds")
    print(f"Adjugate Matrix Inverse: {time_function(matrix.adjugate_inverse):.6f} seconds")
    print("-" * 40)

    matrix = mm.SquareMatrix.generate_random_matrix(100)
    print(f"Testing performance with a {100}x{100} matrix:")
    print(f"Gaussian determinant: {time_function(matrix.gaussian_determinant):.6f} seconds")
    print(f"Gaussian Matrix Inverse: {time_function(matrix.gaussian_inverse):.6f} seconds")
    print(f"Matrix Multiply: {time_function(matrix.multiply,matrix):.6f} seconds")
    print("-" * 40)

    matrix = mm.SquareMatrix.generate_random_matrix(1000)
    print(f"Testing performance with a {1000}x{1000} matrix:")
    print(f"Hadamard Product: {time_function(matrix.hadamard_product,matrix):.6f} seconds")
    print(f"Add Matrices: {time_function(matrix.add, matrix):.6f} seconds")
    print(f"Subtract Matrices: {time_function(matrix.subtract,matrix):.6f} seconds")
    print("-" * 40)
def generate_illegal_matrices():
    """生成非法矩阵"""
    illegal_matrices = [
        [[1, 2], [3, 4], [5]],      # 不规则矩阵（行长度不同）
        [[1, 'a'], [3, 4]],         # 包含非数字元素
        None,                        # None类型
        [[1, 2], [3, 4], []],       # 一行为空
        [[1, 2, 3], [4, 5]]         # 非方阵用于行列式和逆矩阵测试
    ]
    return illegal_matrices
def test_matrix_operations():
    illegal_matrices = generate_illegal_matrices()
    
    for matrix in illegal_matrices:
        print(f"Testing with matrix: {matrix}")
        try:
            mm.Matrix(matrix)
        except Exception as e:
            print("Matrix Exception:", e)
        print("-" * 40)
def test_exceptions():
    # 测试非方阵
    matrix = mm.Matrix.generate_random_matrix(2, 3)  # 2行3列的矩阵
    print("Testing non-square matrix:")
    try:
        mm.SquareMatrix(matrix.data)
    except ValueError as e:
        print(f"成功捕获异常: {e}")

    # 测试形状不同的矩阵
    matrix_a = mm.Matrix.generate_random_matrix(3, 3)  # 3x3 矩阵
    matrix_b = mm.Matrix.generate_random_matrix(3, 2)  # 3x2 矩阵
    print("Testing matrices of different shapes:")
    try:
        if not matrix_b.hadamard_product(matrix_a):
            raise ValueError("矩阵形状不同")
    except ValueError as e:
        print(f"成功捕获异常: {e}")
    
if __name__ == "__main__":
    test_performance()
