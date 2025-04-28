import boto3
import io
import zipfile
import pandas as pd
from io import RawIOBase

class S3File(RawIOBase):
    """A file-like object for S3 that supports seeking without downloading the entire file."""
    
    def __init__(self, s3_object):
        self.s3_object = s3_object
        self.position = 0
        
        # Get the file size
        self.size = self.s3_object.content_length
        
        # Read the central directory first (located at the end of the ZIP file)
        # Reading the last 64KB is usually sufficient to get the central directory
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
        
        return self.position
    
    def tell(self):
        return self.position
    
    def read(self, size=-1):
        # If size is negative, read until the end of the file
        if size < 0:
            size = self.size - self.position
        
        # Check if the requested data is already in the buffer
        if (self.position >= self._buffer_start and 
            self.position + size <= self._buffer_end):
            start = self.position - self._buffer_start
            data = self._buffer[start:start + size]
            self.position += size
            return data
        
        # If not in buffer, request the specific range from S3
        start = self.position
        end = min(self.position + size - 1, self.size - 1)
        
        if start > end:
            return b""
        
        # Get the specific byte range
        response = self.s3_object.get(Range=f'bytes={start}-{end}')
        data = response['Body'].read()
        
        # Update buffer
        self._buffer = data
        self._buffer_start = start
        self._buffer_end = end + 1
        
        # Update position
        self.position += len(data)
        
        return data

def extract_csv_from_zip_in_s3(bucket_name, zip_key, csv_name):
    """
    Extract a specific CSV file from a ZIP archive in S3 without downloading the entire ZIP.
    
    Args:
        bucket_name: S3 bucket name
        zip_key: Key to the ZIP file in S3
        csv_name: Name of the CSV file within the ZIP to extract
        
    Returns:
        pandas DataFrame containing the CSV data
    """
    # Initialize S3 resource
    s3 = boto3.resource('s3',
        endpoint_url="http://localhost:9000",
        aws_access_key_id="miniotestadmin",
        aws_secret_access_key="miniotestsecret",
        config=boto3.session.Config(signature_version='s3v4')
    )
    
    # Get the S3 object
    s3_object = s3.Object(bucket_name, zip_key)
    
    # Create our custom file-like object
    s3_file = S3File(s3_object)
    
    # Open the ZIP file using our custom file-like object
    with zipfile.ZipFile(s3_file) as zip_file:
        # Print all files in the archive
        print(f"Files in archive: {zip_file.namelist()}")
        
        # Check if our target file exists
        if csv_name not in zip_file.namelist():
            raise FileNotFoundError(f"{csv_name} not found in ZIP archive")
        
        # Read the CSV file directly from the ZIP
        with zip_file.open(csv_name) as file:
            df = pd.read_csv(file)
            
    return df

if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = "default"
    ZIP_KEY = "default/maxheadroom/rocrates/data.zip"
    CSV_NAME = "data/example_data.csv"
    
    # Extract and load the CSV
    try:
        df = extract_csv_from_zip_in_s3(BUCKET_NAME, ZIP_KEY, CSV_NAME)
        
        # Display data
        print("\nSuccessfully extracted CSV data")
        print(f"DataFrame shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Optionally save to local file
        df.to_csv("extracted_data.csv", index=False)
        print(f"\nSaved data to extracted_data.csv")
        
    except Exception as e:
        print(f"Error: {str(e)}")