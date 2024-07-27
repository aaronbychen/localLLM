import mysql.connector

# 数据库配置
DB_CONFIG = {
    'user': 'root',
    'password': '123123123123aA',
    'host': '127.0.0.1',
    'database': 'user_db',
}


def test_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE();")
        result = cursor.fetchone()
        print(f"Successfully connected to database: {result[0]}")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    test_db_connection()
