import boto3
import os

#define the folder containing files to upload and the bucket name
input_folder = 'input_folder'
bucket_name = 'image.upload.group14' 

#initializeingthe s3 client
s3_client = boto3.client('s3')

#iterate over files in imagw folder
for file_name in os.listdir(input_folder):
    # Create file path
    file_path = os.path.join(input_folder, file_name)
    # Check if it is a file and upload file, print success into console
    if os.path.isfile(file_path):
        s3_client.upload_file(file_path, bucket_name, file_name)
        print(f"Upload Successful: {file_path} to {bucket_name}/{file_name}")

