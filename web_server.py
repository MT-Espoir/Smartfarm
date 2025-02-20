from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import tensorflow as tf
from utils import *
from plant_tracker import PlantTracker
import os
import threading
import time
from utils import get_latest_data
from flask_cors import CORS
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
DATA_FILE = r'Data/sensor_data.csv'


MQTT_BROKER = "broker.hivemq.com"  # Hoặc dùng MQTT broker riêng của bạn
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "yolouno/pump"  # Chủ đề MQTT để điều khiển bơm nước

# Kết nối MQTT
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

MODEL_PATH = r'Data/my_model.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

CLASS_LABELS = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy',
    'Rice_leaf___Bacterial_leaf_blight','Rice_leaf___Brown_spot',
]
UPLOAD_FOLDER = os.path.join(os.getcwd(), "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

plant_tracker = PlantTracker()

def sensor_data_emitter():
    while True:
        data = get_latest_data()
        if data:
            socketio.emit('sensor_data', data)
        time.sleep(1)  # Phát dữ liệu mỗi 5 giây

threading.Thread(target=sensor_data_emitter, daemon=True).start()

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/data')
def get_data():
    df = load_data()
    if not df.empty:
        return df.tail(100).to_json(orient='records', date_format='iso')
    return []

@app.route('/data_by_date', methods=['GET'])
def get_data_by_date():
    date = request.args.get('date')
    if not date:
        return jsonify({'success': False, 'message': 'Date is required'}), 400

    try:
        data = get_by_date(date)
        if data:
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'message': 'No data found for the given date'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/send_command', methods=['POST'])
def send_command():
    data = request.get_json()
    action = data.get('action')

    if action in ['pump_on', 'pump_off']:
        mqtt_client.publish(MQTT_TOPIC_COMMAND, action)
        return {'status': 'success'}
    return {'status': 'invalid command'}, 400

@app.route('/get_command')
def get_current_command():
    global current_command
    command = current_command
    current_command = None  # Reset sau khi trả về
    return {'command': command}

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Tạo thư mục lưu ảnh nếu chưa có
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Tạo tên file mới với timestamp để tránh ghi đè
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Lưu file ảnh vào thư mục trên máy
        file.save(file_path)

        # Thực hiện dự đoán
        class_name, prob = predict_image_file(file_path, model, CLASS_LABELS)

        # Trả về kết quả JSON
        return jsonify({"class": class_name, "probability": prob, "saved_path": file_path})
    else:
        return render_template('predict.html')

@app.route('/plant_tracker')
def plant_tracker_page():
    plants = plant_tracker.get_all_plants()
    alerts = plant_tracker.get_alerts()
    return render_template('plant_tracker.html', plants=plants, alerts=alerts)

@app.route('/add_plant', methods=['POST'])
def add_plant():
    data = request.json
    plant_id = plant_tracker.add_plant(
        data['plant_name'],
        int(data['growth_days']),
        data.get('notes', '')
    )
    return jsonify({'success': True, 'plant_id': plant_id})

@app.route('/update_plant_status', methods=['POST'])
def update_plant_status():
    data = request.json
    success = plant_tracker.update_plant_status(
        int(data['plant_id']),
        data['status'],
        data.get('notes', None)
    )
    return jsonify({'success': success})

@app.route('/plant_details/<int:plant_id>')
def plant_details(plant_id):
    status = plant_tracker.get_plant_status(plant_id)
    return render_template('plant_details.html', status=status)

@app.route('/api/plants')
def get_plants():
    plants = plant_tracker.get_all_plants()
    return jsonify({"plants": plants})

@app.route('/delete_plant', methods=['POST'])
def delete_plant():
    data = request.json
    plant_id = data.get('plant_id')
    if plant_id is None:
        return jsonify({'success': False, 'message': 'Plant ID is required'}), 400

    success = plant_tracker.delete_plant(int(plant_id))
    return jsonify({'success': success})

@app.route('/api/alerts')
def get_plant_alerts():
    alerts = plant_tracker.get_alerts()
    return jsonify({"alerts": alerts})

@socketio.on('message')
def handle_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message = msg.get('message', '')
    lower_message = user_message.lower()

    response = "Xin lỗi, tôi không hiểu câu hỏi của bạn."

    # Xử lý các câu hỏi mặc định theo nội dung
    if "nhiệt độ hôm nay" in lower_message:
        avg_temp = get_today_avg_temperature()
        lasted_temp = get_latest_data()['temperature']
        if avg_temp is not None:
            response = f"Nhiệt độ trung bình hôm nay là {avg_temp:.2f}°C.\n"\
                       f"Nhiệt độ hiện tại là {lasted_temp:.2f}°C"
        else:
            response = "Không có dữ liệu nhiệt độ hôm nay."
    elif "độ ẩm hôm nay" in lower_message:
        avg_humidity = get_today_avg_humidity()
        lasted_humidity = get_latest_data()['humidity']
        if avg_humidity is not None:
            response = f"Độ ẩm trung bình hôm nay là {avg_humidity:.2f}%.\n"\
                       f"Độ ẩm độ hiện tại là {lasted_humidity:.2f}%"
        else:
            response = "Không có dữ liệu độ ẩm hôm nay."
    elif "bơm nước hôm nay" in lower_message:
        pump_count = get_today_pump_count()
        response = f"Hệ thống đã bơm nước {pump_count} lần hôm nay."
    elif "hệ thống có ổn định" in lower_message:
        # Gọi hàm kiểm tra độ ổn định của hệ thống
        response = check_system_stability()
    elif "ngày thu hoạch" in lower_message:
        plants = plant_tracker.get_all_plants()
        harvest_info = [
            f"{plant['plant_name']} dự kiến thu hoạch vào {plant['expected_harvest_date']}"
            for plant in plants
        ]
        response = "\n".join(harvest_info) if harvest_info else "Không có dữ liệu cây trồng."

    elif "cây nào sắp thu hoạch" in lower_message:
        alerts = plant_tracker.get_alerts()
        response = "\n".join([alert['message'] for alert in alerts]) if alerts else "Không có cây nào sắp thu hoạch."
    # Gửi tin nhắn người dùng và phản hồi từ hệ thống
    emit('chat_message', {
        'timestamp': timestamp,
        'username': msg.get('username', 'Anonymous'),
        'message': user_message
    }, broadcast=True)
    emit('chat_message', {
        'timestamp': timestamp,
        'username': 'ChatBot',
        'message': response
    }, broadcast=True)


@socketio.on('connect')
def handle_connect():
    # Có thể gửi lịch sử chat nếu cần, hiện tại để rỗng
    emit('chat_history', [])


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
