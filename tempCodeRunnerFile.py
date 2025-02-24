@app.route('/api/sensor_data', methods=['GET'])
def get_sensor_data():
    data = load_data()
    response = jsonify(data.to_dict(orient='records'))
    return response