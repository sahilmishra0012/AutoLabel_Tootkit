'''
Script to read data from GCS bucket, write data to GCS bucket and
list files in a directory of the bucket.
'''
import warnings
from google.cloud import storage
warnings.filterwarnings("ignore")


def download_blob(bucket_name, source_blob_name, destination_file_name):

    '''Function to download blobs from GCP GCS bucket

        Parameters:
            bucket_name             - The name of the bucket to be instantiated.
            source_blob_name        - The blob resource to download.
            destination_file_name   - A file handle to which to write the blob’s data.
        Return Value:
            None
    '''

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def upload_blob(bucket_name, source_file_name, destination_blob_name):

    '''Function to upload blobs to GCP GCS bucket directory

        Parameters:
            bucket_name             - The name of the bucket to be instantiated.
            source_blob_name        - The blob resource to upload.
            destination_blob_name   - A file handle to which to write the blob’s data.
        Return Value:
            None
    '''

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def get_blobs(bucket_name, directory_path):

    '''Function to return blobs from a particular GCP GCS bucket directory

        Parameters:
            bucket_name             - The name of the bucket to be instantiated.
            directory_path          - The directory whose blobs are to be returned
        Return Value:
            List of files in a particular GCS GCP GCS bucket directory
    '''

    storage_client = storage.Client()

    files = []
    for blob in storage_client.list_blobs(bucket_name, prefix=directory_path):
        files.append(blob.name)

    return files
