import mysql.connector
from mysql.connector import errorcode
import bcrypt
import re

# 数据库配置
DB_CONFIG = {
    'user': 'root',
    'password': 'zxc7777777',
    'host': 'localhost',
    'database': 'test_db'
}

# 存储用户信息的字典
user_data = {}

def validate_email(email):
    """ 验证电子邮件格式 """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def validate_password(password):
    """ 验证密码复杂度 """
    password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&;:"#_]{8,}$'
    return re.match(password_pattern, password) is not None

def register_user():
    """ 注册用户的函数 """
    while True:
        username = input("请输入用户名: ")
        # 检查用户名是否已存在
        if username in user_data:
            print("用户名已存在，请重新输入。")
        else:
            break

    email = input("请输入电子邮件: ")
    while not validate_email(email):
        print("电子邮件格式不正确，请重新输入。")
        email = input("请输入电子邮件: ")

    password = input("请输入密码: ")
    while not validate_password(password):
        print("密码不符合复杂度要求，请重新输入。")
        password = input("请输入密码: ")

    # 将密码哈希化
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # 存储用户信息到字典
    user_data[username] = {
        'email': email,
        'password': hashed_password.decode('utf-8')
    }

    # 将用户信息存入 MySQL 数据库
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 创建用户表（如果尚不存在）
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        )
        '''
        cursor.execute(create_table_query)

        # 插入用户信息
        insert_query = '''
        INSERT INTO users (username, email, password) 
        VALUES (%s, %s, %s)
        '''
        cursor.execute(insert_query, (username, email, user_data[username]['password']))

        # 提交更改
        conn.commit()
        print("用户注册成功，信息已存入数据库。")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DUP_ENTRY:
            print("用户名或电子邮件已存在。")
        else:
            print(f"发生错误: {err}")
    finally:
        cursor.close()
        conn.close()

def login_user():
    """ 登录用户的函数 """
    username = input("请输入用户名: ")
    password = input("请输入密码: ")

    # 连接到 MySQL 数据库
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 查询用户信息
        query = "SELECT password FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        result = cursor.fetchone()

        # 检查用户是否存在并验证密码
        if result:
            stored_password = result[0].encode('utf-8')  # 从数据库中获取的密码
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                print("登录成功！")
            else:
                print("密码错误。")
        else:
            print("用户名不存在。")
    except mysql.connector.Error as err:
        print(f"发生错误: {err}")
    finally:
        cursor.close()
        conn.close()

# 示例调用
if __name__ == "__main__":
    while True:
        action = input("请选择操作：1. 注册用户 2. 登录用户 3. 退出\n")
        if action == '1':
            register_user()
        elif action == '2':
            login_user()
        elif action == '3':
            print("退出程序。")
            break
        else:
            print("无效的选择，请重新输入。")
