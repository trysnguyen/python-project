import matplotlib.pyplot as plt
from sql import get_emotion_data


def plot_emotion_data():
    # Lấy dữ liệu từ database MySQL
    data = get_emotion_data()

    # Đếm số lần xuất hiện của mỗi cảm xúc
    emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    emotion_counts = {emotion: 0 for emotion in emotions}

    for row in data:
        emotion = row[1]  # Cảm xúc từ cột thứ hai
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1

    # Vẽ đồ thị
    plt.bar(emotion_counts.keys(), emotion_counts.values())
    plt.xlabel('Cảm xúc')
    plt.ylabel('Số lần xuất hiện')
    plt.title('Tần suất các cảm xúc')
    plt.show()


# Gọi hàm để vẽ đồ thị
plot_emotion_data()
