from flask import Flask, request, jsonify
from model_utils import predict_health_a
from model_utils_b import predict_health_b

app = Flask(__name__)

@app.route('/predict_a', methods=['POST'])
def predict_a():
    """API cho mô hình A."""
    try:
        input_data = request.json
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        probability, advice = predict_health_a(input_data)
        return jsonify({"probability": probability, "advice": advice}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_b', methods=['POST'])
def predict_b():
    """API cho mô hình B."""
    try:
        input_data = request.json
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        probability, advice = predict_health_b(input_data)
        return jsonify({"probability": probability, "advice": advice}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
