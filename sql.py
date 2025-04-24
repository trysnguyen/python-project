import mysql.connector
from mysql.connector import Error

# Kết nối đến MySQL
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',          # Địa chỉ server MySQL
            database='emotion_recognition',  # Tên cơ sở dữ liệu
            user='example_user',               # Tên người dùng
            password='Banhbun@123'        # Mật khẩu người dùng
        )
        if connection.is_connected():
            print("Kết nối thành công!")
        return connection
    except Error as e:
        print(f"Lỗi kết nối MySQL: {e}")
        return None

# Lưu cảm xúc vào cơ sở dữ liệu
def save_emotion_to_db(emotion):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO emotion_data (emotion) VALUES (%s)", (emotion,))
        connection.commit()
        cursor.close()
        connection.close()

# Lấy dữ liệu cảm xúc từ cơ sở dữ liệu
def get_emotion_data():
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM emotion_data")
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    return []
