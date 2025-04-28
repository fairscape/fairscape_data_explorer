# s3_utils.py
import io
import zipfile
import os
import boto3
from io import RawIOBase
from botocore.exceptions import ClientError, NoCredentialsError
from config import MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, AWS_REGION

class S3File(RawIOBase):
    def __init__(self, s3_client, bucket, key):
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self.position = 0

        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=self.key)
            self.size = response['ContentLength']
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}") from e
            else:
                raise
        
        self._buffer = b""
        self._buffer_start = 0
        self._buffer_end = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset

        # Clamp position to valid range
        self.position = max(0, min(self.position, self.size))
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size == 0:
            return b""
        
        if self.position >= self.size:
            return b"" # End of file reached

        # If size is negative or requests more than available, read until the end
        if size < 0 or self.position + size > self.size:
            size = self.size - self.position

        # Check if the requested data is already in the buffer
        buffer_available_start = self.position - self._buffer_start
        if buffer_available_start >= 0 and buffer_available_start + size <= len(self._buffer):
            data = self._buffer[buffer_available_start : buffer_available_start + size]
            self.position += len(data)
            return data

        # Determine the range to fetch from S3
        fetch_start = self.position
        # Request a slightly larger chunk for potential future reads, but respect limits
        fetch_size = max(size, 65536) # Read at least 64KB or requested size
        fetch_end = min(fetch_start + fetch_size - 1, self.size - 1)

        if fetch_start > fetch_end:
            return b""

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=self.key,
                Range=f'bytes={fetch_start}-{fetch_end}'
            )
            new_data = response['Body'].read()
        except ClientError as e:
             raise IOError(f"Error reading from S3: {e}") from e

        # Update buffer
        self._buffer = new_data
        self._buffer_start = fetch_start
        self._buffer_end = fetch_start + len(new_data)

        # Return the requested portion from the new buffer
        data_to_return = self._buffer[:size]
        self.position += len(data_to_return)
        return data_to_return


def extract_from_zip_in_s3(s3_client, bucket, key, file_path_in_zip):
    s3_file = S3File(s3_client, bucket, key)

    try:
        with zipfile.ZipFile(s3_file, 'r') as zip_file:
            file_list = zip_file.namelist()

            if file_path_in_zip in file_list:
                with zip_file.open(file_path_in_zip) as file:
                    return file.read()
            else:
                # Fallback: Try finding by basename if exact path fails
                target_filename = os.path.basename(file_path_in_zip)
                matching_files = [f for f in file_list if os.path.basename(f) == target_filename]
                if matching_files:
                     with zip_file.open(matching_files[0]) as file:
                        return file.read()

            raise FileNotFoundError(f"File '{file_path_in_zip}' not found in ZIP archive s3://{bucket}/{key}. Available files: {file_list}")
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Error opening ZIP file s3://{bucket}/{key}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during ZIP extraction from s3://{bucket}/{key}: {e}") from e


def get_s3_client():
     try:
        s3 = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name=AWS_REGION,
            config=boto3.session.Config(signature_version='s3v4')
        )
        # Test connection by listing buckets (optional, requires permissions)
        # s3.list_buckets() 
        return s3
     except NoCredentialsError:
         print("AWS credentials not found. Ensure MINIO_ACCESS_KEY and MINIO_SECRET_KEY are set.")
         raise
     except Exception as e:
         print(f"Error initializing S3 client: {e}")
         raise