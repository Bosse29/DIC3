#importing libraries
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
import time

#we're using Flask Framework to create the API
app = Flask(__name__)

#loading up the configuration we were provided
cfg_path = 'yolov3-tiny.cfg'
weights_path = 'yolov3-tiny.weights'
network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = network.getLayerNames()
output_layers = [layer_names[i - 1] for i in network.getUnconnectedOutLayers()]

with open('coco.names', 'r') as file:
    class_labels = file.read().strip().split("\n")

@app.route('/api/object_detection', methods=['POST'])
def detect_objects():
    #expected data format: 'id' and 'image_data' (base64)
    data = request.get_json()
    image_id = data.get('id')
    base64_image = data.get('image_data')

    #image processing
    image_bytes = base64.b64decode(base64_image)
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    #resizing every picture to the same, expected size
    processed_img = cv2.resize(image, (416, 416), interpolation=cv2.INTER_CUBIC)

    #converting to expected file type
    blob = cv2.dnn.blobFromImage(processed_img, 0.00392, (416, 416), swapRB=True, crop=False)
    
    network.setInput(blob)
    
    #time starts
    start_time = time.time()

    #inference starts, we feed the image to the neural network
    outputs = network.forward(output_layers)

    #time stops
    end_time = time.time()

    inference_duration = end_time - start_time
    
    detected_objects = []
    #for each output(detections from different layers)
    for output in outputs:
        #every detection we have found
        for detection in output:
            #check that we have a correct format for detection (Bounding box coordinates are the first 4 elements)
            if len(detection) > 5:
                class_scores = detection[5:]
                highest_class_id = np.argmax(class_scores)
                confidence_score = class_scores[highest_class_id]
                #we store our confident guesses
                if confidence_score > 0.4:
                    detected_objects.append({
                        "label": class_labels[highest_class_id],
                        "accuracy": float(confidence_score)
                })
    return jsonify({"id": image_id, "objects": detected_objects, "inference_time": inference_duration})

if __name__ == '__main__':
    app.run(port=5000)









