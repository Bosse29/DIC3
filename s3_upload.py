import boto3
import os

# Define the folder containing files to upload and the bucket name
input_folder = 'input_folder'  # Replace with your folder path
bucket_name = 'image.upload.group14'  # Replace with your bucket name

# Initialize the S3 client
s3_client = boto3.client('s3')

# Iterate over files in the specified folder
for file_name in os.listdir(input_folder):
    # Create the full file path
    file_path = os.path.join(input_folder, file_name)
    # Check if it's a file (and not a directory)
    if os.path.isfile(file_path):
        s3_client.upload_file(file_path, bucket_name, file_name)
        print(f"Upload Successful: {file_path} to {bucket_name}/{file_name}")

