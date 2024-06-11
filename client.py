import requests
import base64
import uuid
import sys
import os
import json

if __name__ == '__main__':
    images = sys.argv[1]
    endpoint = sys.argv[2]
    total_time = 0
    total_img = 0

    for file in os.listdir(images):
        image_path = os.path.join(images, file)
        with open(image_path, "rb") as file:
            image_data = base64.b64encode(file.read()).decode('utf-8')
        image_id = str(uuid.uuid4())
        data = {"id": image_id, "image_data": image_data}
        response = requests.post(endpoint, json=data)
        result = response.json()
        total_time = (total_time + result['inference_time'])
        total_img = total_img + 1
        print(json.dumps(result))

    averageTime = total_time / total_img
    print(f"Average inference time: {averageTime:.4f} seconds")


