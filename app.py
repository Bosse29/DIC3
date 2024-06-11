from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import uuid
import traceback

app = Flask(__name__)

# Load YOLOv3-tiny model
config_path = 'yolov3-tiny.cfg'
weights_path = 'yolov3-tiny.weights'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load COCO labels
with open('coco.names', 'r') as f:
    labels = f.read().strip().split("\n")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the POST endpoint for object detection
@app.route('/api/object_detection', methods=['POST'])
def detect_objects():
    try:
        data = request.get_json()
        
        # Extract the ID and base64-encoded image data
        image_id = data.get('id', str(uuid.uuid4()))
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode the base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            np_image = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding image: {e}")
            traceback.print_exc()
            return jsonify({"error": "Invalid base64 data"}), 400
        
        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Run the forward pass
        detections = net.forward(output_layers)
        
        # Process detections
        height, width = img.shape[:2]
        objects = []

        for output in detections:
            for detection in output:
                if len(detection) > 5:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # confidence threshold
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        objects.append({
                            "label": labels[class_id],
                            "accuracy": float(confidence)
                        })

        # Return the response
        return jsonify({"id": image_id, "objects": objects})
    except Exception as e:
        print(f"Error during object detection: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during object detection: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # Running on port 5000




