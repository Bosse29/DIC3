import requests
import base64
import os
import uuid
import sys

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def upload_image(image_path, endpoint):
    image_id = str(uuid.uuid4())
    image_data = encode_image(image_path)
    
    payload = {
        "id": image_id,
        "image_data": image_data
    }
    
    response = requests.post(endpoint, json=payload)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Failed to decode JSON response for image {image_id}")
        print(f"Response content: {response.content}")
        return None

if __name__ == '__main__':
    input_folder = sys.argv[1]
    endpoint = sys.argv[2]
    
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        result = upload_image(image_path, endpoint)
        if result:
            print(result)
