#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import boto3
import cv2
import time
import numpy as np
from botocore.exceptions import NoCredentialsError, ClientError
from decimal import Decimal

# Initialize S3 and DynamoDB clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('object_detection')

cfg_path = '/tmp/yolov3-tiny.cfg'
weights_path = '/tmp/yolov3-tiny.weights'
names_path = '/tmp/coco.names'

def download_yolo_files():
    try:
        print("Downloading YOLO files...")
        s3_client.download_file('config.files.group14', 'yolov3-tiny.cfg', cfg_path)
        s3_client.download_file('config.files.group14', 'yolov3-tiny.weights', weights_path)
        s3_client.download_file('config.files.group14', 'coco.names', names_path)
        print("YOLO files downloaded.")
    except ClientError as e:
        print(f"Error downloading YOLO files: {e}")

def load_yolo_network():
    print("Loading YOLO network...")
    network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = network.getLayerNames()
    output_layers = [layer_names[i - 1] for i in network.getUnconnectedOutLayers()]
    with open(names_path, 'r') as file:
        class_labels = file.read().strip().split("\n")
    print("YOLO network loaded.")
    return network, output_layers, class_labels

def process_image(bucket_name, object_key, network, output_layers, class_labels):
    # Download image from S3
    image_path = f'/tmp/{object_key}'
    start_download_time = time.time()
    s3_client.download_file(bucket_name, object_key, image_path)
    end_download_time = time.time()
    download_duration = end_download_time - start_download_time

    print(f"Image downloaded to {image_path}, download duration: {download_duration}")

    # Read the image
    image = cv2.imread(image_path)
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
                if confidence_score > 0.5:
                    detected_objects.append({
                        "label": class_labels[highest_class_id],
                        "accuracy": Decimal(str(confidence_score))
                    })

    # Store results in DynamoDB
    item = {
        'image_ID': object_key,
        'DetectedObjects': detected_objects,
        'InferenceTime': Decimal(str(inference_duration)),
        'TransferDuration': Decimal(str(download_duration)),
        'S3Url': f's3://{bucket_name}/{object_key}'
    }

    try:
        table.put_item(Item=item)
        print(f"Successfully put item in DynamoDB: {object_key}")
    except ClientError as e:
        print(f"Failed to put item in DynamoDB: {e}")

def lambda_handler(event, context):
    print("Lambda function started")

    # Download YOLO files if not already present
    download_yolo_files()

    network, output_layers, class_labels = load_yolo_network()

    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']

        try:
            process_image(bucket_name, object_key, network, output_layers, class_labels)
        except Exception as e:
            print(f"Error processing {object_key} from {bucket_name}: {e}")

    print("Lambda function completed")
    return {
        'statusCode': 200,
        'body': json.dumps('Object detection completed.')
    }

