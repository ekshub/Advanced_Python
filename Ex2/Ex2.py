def selection_sort(arr):
    """
    对给定的列表使用选择排序算法进行升序排序。
    
    参数:
    arr (list): 待排序的列表
    
    返回:
    list: 排序后的列表
    """
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def get_valid_input(user_input):
    """
    获取用户输入，并将其分为数值和字符型数据，确保输入数据不混合。
    
    参数:
    user_input (str): 用户输入的字符串
    
    返回:
    tuple: (排序后的数值列表, 排序后的字符列表)
    """
    if not user_input.strip():
        # 如果输入为空或仅包含空白字符，返回空列表
        return ([], [])
    
    parts = user_input.split()
    numbers = []
    strings = []
    
    for part in parts:
        try:
            number = float(part)
            numbers.append(number)
        except ValueError:
            strings.append(part)
    
    # 检查是否有混合的情况
    if numbers and strings:
        print("输入数据混合了数值和字符，请分开输入。")
        return ([], [])
    
    return (selection_sort(numbers), selection_sort(strings))

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])
    
    return merge(left_half, right_half)

def merge(left, right):
    sorted_arr = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_arr.append(left[i])
            i += 1
        else:
            sorted_arr.append(right[j])
            j += 1
            
    # Append remaining elements
    sorted_arr.extend(left[i:])
    sorted_arr.extend(right[j:])
    
    return sorted_arr

def test():
    test_cases = [
        "3 1 4 1 5 9 2 6 5 3 5", 
        "banana apple cherry date", 
        "", 
        "3 1 4 banana 1 5 apple 9",  
        "banana 3 apple 1 cherry 4",  
        "-3 0 2 -1 5 -6 4", 
        "Zebra apple Banana", 
        "7",  
        "hello", 
        "10.5 apple 4 7.2 banana"  
    ]

    for i, test_case in enumerate(test_cases, 1):
        result = get_valid_input(test_case)
        print(f"Test Case {i}:")
        print(f"Input: {test_case}")
        print(f"Output: {result}\n")

if __name__ == "__main__":
    # Uncomment to run the test cases
    # test()
    
    arr = input("请输入一组以空格分隔的数据：")
    numbers, strings = get_valid_input(arr)

    # 如果需要对输入数据的数字部分进行排序
    if numbers:
        print("选择排序后的数值列表:", selection_sort(numbers))
        print("归并排序后的数值列表:", merge_sort(numbers))
    if strings:
        print("选择排序后的字符列表:", selection_sort(strings))
        print("归并排序后的字符列表:", merge_sort(strings))
