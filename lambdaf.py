import json
import boto3
import cv2
import time
import numpy as np
from decimal import Decimal

#using AWS API boto3 to access s3 and dynamoDB

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('object_detection')

#creating temporal paths

cfg_path = '/tmp/yolov3-tiny.cfg'
weights_path = '/tmp/yolov3-tiny.weights'
names_path = '/tmp/coco.names'

#defining the module for loading in the yolo configs, weights and coco class labels

def download_yolo_files():
    s3_client.download_file('config.files.group14', 'yolov3-tiny.cfg', cfg_path)
    s3_client.download_file('config.files.group14', 'yolov3-tiny.weights', weights_path)
    s3_client.download_file('config.files.group14', 'coco.names', names_path)
    
#creating the yolo network with the provided files

def load_yolo_network():
    network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = network.getLayerNames()
    output_layers = [layer_names[i - 1] for i in network.getUnconnectedOutLayers()]
    with open(names_path, 'r') as file:
        class_labels = file.read().strip().split("\n")
    return network, output_layers, class_labels


def process_image(bucket_name, object_key, network, output_layers, class_labels):
    
    #get image from s3 bucket and take transfer tie
    image_path = f'/tmp/{object_key}'

    start_download_time = time.time()
    s3_client.download_file(bucket_name, object_key, image_path)
    end_download_time = time.time()
    download_duration = end_download_time - start_download_time
    #image processing, resizing to expected size and converting to expected file type
    image = cv2.imread(image_path)
    processed_img = cv2.resize(image, (416, 416), interpolation=cv2.INTER_CUBIC)
    blob = cv2.dnn.blobFromImage(processed_img, 0.00392, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    #time starts
    start_time = time.time()
    #inference starts, we feed the image to the neural network
    outputs = network.forward(output_layers)
    #time ends
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
                if confidence_score > 0.5:
                    detected_objects.append({
                        "label": class_labels[highest_class_id],
                        "accuracy": Decimal(str(confidence_score))
                    })
    #put the created items into the dynamoDB table
    table.put_item(Item={
        'image_ID': object_key,
        'DetectedObjects': detected_objects,
        'InferenceTime': Decimal(str(inference_duration)),
        'TransferDuration': Decimal(str(download_duration)),
        'S3Url': f's3://{bucket_name}/{object_key}'
    })

    #handling the lambda function
def lambda_handler(event, context):
    #getting the yolo files and class labels by calling the module
    download_yolo_files()
    network, output_layers, class_labels = load_yolo_network()
    #processing every image by calling the created modules
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        process_image(bucket_name, object_key, network, output_layers, class_labels)
