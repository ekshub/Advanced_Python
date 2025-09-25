import mysql.connector
from mysql.connector import errorcode
import bcrypt
import re
import json
import logging
import sys
import os

# 配置日志记录
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

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
    """验证电子邮件格式"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = re.match(email_pattern, email) is not None
    logging.debug(f"验证邮箱 '{email}' 格式: {'有效' if is_valid else '无效'}")
    return is_valid

def validate_password(password):
    """验证密码复杂度"""
    password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&;:"#_])[A-Za-z\d@$!%*?&;:"#_]{8,}$'
    is_valid = re.match(password_pattern, password) is not None
    logging.debug(f"验证密码复杂度: {'符合' if is_valid else '不符合'}")
    return is_valid

def load_user_data():
    """从文件加载用户数据"""
    global user_data
    try:
        if not os.path.exists('users_data.txt'):
            logging.info("用户数据文件不存在，创建新的文件。")
            with open('users_data.txt', 'w', encoding='utf-8') as f:
                json.dump({}, f)
        with open('users_data.txt', 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        logging.info("已从文件加载用户数据。")
    except json.JSONDecodeError as e:
        logging.error(f"用户数据文件格式错误: {e}")
        user_data = {}
    except Exception as e:
        logging.exception(f"读取用户数据时发生错误: {e}")
        user_data = {}

def save_user_data():
    """将用户数据保存到文件"""
    try:
        with open('users_data.txt', 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=4)
        logging.info("用户数据已保存到文件。")
    except Exception as e:
        logging.exception(f"保存用户数据时发生错误: {e}")

def connect_db():
    """连接数据库，返回连接和游标"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        logging.debug("成功连接到数据库。")
        return conn, cursor
    except mysql.connector.Error as err:
        logging.exception(f"数据库连接失败: {err}")
        sys.exit("无法连接到数据库，请检查配置。")

def register_user():
    """注册用户的函数"""
    try:
        username = input("请输入用户名: ")
        while username in user_data:
            print("用户名已存在，请重新输入。")
            username = input("请输入用户名: ")
        
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
        conn, cursor = connect_db()

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
        logging.info(f"新用户注册成功: {username}")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DUP_ENTRY:
            print("用户名或电子邮件已存在。")
        else:
            print(f"发生错误: {err}")
        logging.exception(f"MySQL 错误: {err}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        logging.exception(f"注册用户时发生异常: {e}")
    finally:
        cursor.close()
        conn.close()

def login_user():
    """登录用户的函数"""
    try:
        username = input("请输入用户名: ")
        password = input("请输入密码: ")

        if username not in user_data:
            print("用户名不存在。")
            return

        stored_password = user_data[username]['password'].encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
            print("登录成功！")
            logging.info(f"用户登录成功: {username}")
        else:
            print("密码错误。")
            logging.warning(f"用户登录失败（密码错误）: {username}")
    except Exception as e:
        print(f"发生错误: {e}")
        logging.exception(f"登录用户时发生异常: {e}")

def change_password():
    """修改用户密码的函数"""
    username = input("请输入您的用户名: ")

    if username not in user_data:
        print("用户名不存在。")
        return

    current_password = input("请输入当前密码: ")
    stored_password = user_data[username]['password'].encode('utf-8')
    if not bcrypt.checkpw(current_password.encode('utf-8'), stored_password):
        print("当前密码不正确。")
        logging.warning(f"用户修改密码失败（当前密码错误）: {username}")
        return

    new_password = input("请输入新密码: ")
    while not validate_password(new_password):
        print("密码不符合复杂度要求，请重新输入。")
        new_password = input("请输入新密码: ")

    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    user_data[username]['password'] = hashed_password.decode('utf-8')

    # 更新数据库
    conn = None
    cursor = None
    try:
        conn, cursor = connect_db()

        update_query = "UPDATE users SET password = %s WHERE username = %s"
        cursor.execute(update_query, (user_data[username]['password'], username))
        conn.commit()
        print("密码已成功更新。")
        logging.info(f"用户密码更新成功: {username}")
    except mysql.connector.Error as err:
        print(f"发生错误: {err}")
        logging.exception(f"MySQL 错误: {err}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        logging.exception(f"修改密码时发生异常: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

def update_email():
    """更新用户电子邮件的函数"""
    try:
        username = input("请输入您的用户名: ")

        if username not in user_data:
            print("用户名不存在。")
            return

        password = input("请输入您的密码: ")
        stored_password = user_data[username]['password'].encode('utf-8')
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password):
            print("密码不正确。")
            logging.warning(f"用户更新邮箱失败（密码错误）: {username}")
            return

        new_email = input("请输入新的电子邮件: ")
        while not validate_email(new_email):
            print("电子邮件格式不正确，请重新输入。")
            new_email = input("请输入新的电子邮件: ")

        user_data[username]['email'] = new_email

        # 更新数据库
        conn, cursor = connect_db()

        update_query = "UPDATE users SET email = %s WHERE username = %s"
        cursor.execute(update_query, (new_email, username))
        conn.commit()
        print("电子邮件已成功更新。")
        logging.info(f"用户邮箱更新成功: {username}")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DUP_ENTRY:
            print("该电子邮件已被使用。")
            logging.warning(f"邮箱更新失败，邮箱已存在: {new_email}")
        else:
            print(f"发生错误: {err}")
            logging.exception(f"MySQL 错误: {err}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        logging.exception(f"更新邮箱时发生异常: {e}")
    finally:
        cursor.close()
        conn.close()

def delete_account():
    """删除用户账户的函数"""
    try:
        username = input("请输入您的用户名: ")

        if username not in user_data:
            print("用户名不存在。")
            return

        password = input("请输入您的密码: ")
        stored_password = user_data[username]['password'].encode('utf-8')
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password):
            print("密码不正确。")
            logging.warning(f"用户删除账户失败（密码错误）: {username}")
            return

        confirm = input("您确定要删除您的账户吗？此操作无法撤销。(y/n): ")
        if confirm.lower() != 'y':
            print("取消删除账户。")
            logging.info(f"用户取消删除账户: {username}")
            return

        # 删除用户数据
        del user_data[username]

        # 从数据库中删除用户
        conn, cursor = connect_db()

        delete_query = "DELETE FROM users WHERE username = %s"
        cursor.execute(delete_query, (username,))
        conn.commit()
        print("账户已成功删除。")
        logging.info(f"用户账户已删除: {username}")
    except mysql.connector.Error as err:
        print(f"发生错误: {err}")
        logging.exception(f"MySQL 错误: {err}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        logging.exception(f"删除账户时发生异常: {e}")
    finally:
        cursor.close()
        conn.close()

def backup_user_data():
    """备份用户数据到指定文件"""
    try:
        backup_file = input("请输入备份文件名（例如 backup.json）: ")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=4)
        print(f"用户数据已备份到 {backup_file}")
        logging.info(f"用户数据已备份到文件: {backup_file}")
    except Exception as e:
        print(f"备份用户数据时发生错误: {e}")
        logging.exception(f"备份用户数据时发生异常: {e}")

def restore_user_data():
    """从备份文件恢复用户数据"""
    global user_data
    try:
        backup_file = input("请输入要恢复的备份文件名（例如 backup.json）: ")
        if not os.path.exists(backup_file):
            print("备份文件不存在。")
            logging.error(f"备份文件不存在: {backup_file}")
            return
        with open(backup_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        print(f"用户数据已从 {backup_file} 恢复。")
        logging.info(f"用户数据已从文件恢复: {backup_file}")
    except json.JSONDecodeError as e:
        print(f"备份文件格式错误: {e}")
        logging.exception(f"备份文件格式错误: {e}")
    except Exception as e:
        print(f"恢复用户数据时发生错误: {e}")
        logging.exception(f"恢复用户数据时发生异常: {e}")

def export_user_data():
    """将用户数据导出为CSV文件"""
    try:
        export_file = input("请输入导出文件名（例如 users.csv）: ")
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write('用户名,电子邮件\n')
            for username, info in user_data.items():
                f.write(f"{username},{info['email']}\n")
        print(f"用户数据已导出到 {export_file}")
        logging.info(f"用户数据已导出到文件: {export_file}")
    except Exception as e:
        print(f"导出用户数据时发生错误: {e}")
        logging.exception(f"导出用户数据时发生异常: {e}")

if __name__ == "__main__":
    load_user_data()  # 程序启动时加载用户数据
    print(os.getcwd())
    try:
        while True:
            action = input(
                "\n请选择操作：\n"
                "1. 注册用户\n"
                "2. 登录用户\n"
                "3. 修改密码\n"
                "4. 更新电子邮件\n"
                "5. 删除账号\n"
                "6. 备份用户数据\n"
                "7. 恢复用户数据\n"
                "8. 导出用户数据\n"
                "9. 退出\n"
                "请输入数字选择操作: "
            )
            if action == '1':
                register_user()
            elif action == '2':
                login_user()
            elif action == '3':
                change_password()
            elif action == '4':
                update_email()
            elif action == '5':
                delete_account()
            elif action == '6':
                backup_user_data()
            elif action == '7':
                restore_user_data()
            elif action == '8':
                export_user_data()
            elif action == '9':
                print("退出程序。")
                logging.info("程序正常退出。")
                break
            else:
                print("无效的选择，请重新输入。")
                logging.warning(f"无效的菜单选择: {action}")
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在保存用户数据并退出。")
        logging.info("程序被用户中断。")
    finally:
        save_user_data()  # 程序结束时保存用户数据
