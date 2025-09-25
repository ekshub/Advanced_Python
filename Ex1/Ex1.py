def main():
    # 提示用户输入姓名
    name = input("请输入您的姓名：")
    
    # 提示用户输入一个数字
    number_str = input("请输入一个数字：")
    
    # 检查输入的是否为数字
    if number_str.isdigit():
        number = int(number_str)
        # 对数字进行平方计算
        result = number ** 2
        # 输出结果
        print(f"您好，{name}！您输入的数字的平方是：{result}")
    else:
        print("输入的不是有效的数字，请重新运行程序。")

# 判断是否作为主程序运行
if __name__ == "__main__":
    main()