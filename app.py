import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

cfg_path = 'yolov3-tiny.cfg'
weights_path = 'yolov3-tiny.weights'
network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = network.getLayerNames()
output_layers = [layer_names[i - 1] for i in network.getUnconnectedOutLayers()]

with open('coco.names', 'r') as file:
    class_labels = file.read().strip().split("\n")

@app.route('/api/object_detection', methods=['POST'])
def detect_objects():
    data = request.get_json()
    image_id = data.get('id')
    base64_image = data.get('image_data')
    image_bytes = base64.b64decode(base64_image)
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    processed_img = cv2.resize(image, (416, 416), interpolation=cv2.INTER_CUBIC)
    blob = cv2.dnn.blobFromImage(processed_img, 0.00392, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    start_time = time.time()
    
    outputs = network.forward(output_layers)
    
    end_time = time.time()
    inference_duration = end_time - start_time
    
    detected_objects = []
    for output in outputs:
        for detection in output:
            if len(detection) > 5:
                class_scores = detection[5:]
                highest_class_id = np.argmax(class_scores)
                confidence_score = class_scores[highest_class_id]
                if confidence_score > 0.4:
                    detected_objects.append({
                        "label": class_labels[highest_class_id],
                        "accuracy": float(confidence_score)
                })
    return jsonify({"id": image_id, "objects": detected_objects, "inference_time": inference_duration})

if __name__ == '__main__':
    app.run(port=5000)









