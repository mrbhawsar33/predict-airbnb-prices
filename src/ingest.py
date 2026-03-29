import s3fs
import pandas as pd

# # Initialize the S3 FileSystem, uses default AWS CLI profile
# fs = s3fs.S3FileSystem(anon=False)

# # List files in S3 bucket to verify the connection
# s3_path = "staywise-data-bucket/airbnb/raw_data/"
# print(fs.ls(s3_path))

def fetch_s3_data(bucket_dir, file_name):
    """
    Connects to S3 using s3fs and loads a CSV into a Pandas DataFrame.
    """
    s3_path = f"s3://{bucket_dir}/{file_name}"
    fs = s3fs.S3FileSystem(anon=False)

    try:
        print(f"Attempting to access: {s3_path}")
        with fs.open(s3_path, mode='rb') as f:
            df = pd.read_csv(f)
        print("Data loaded successfully!")
        return df
    
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found in bucket '{bucket_dir}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None

if __name__ == "__main__":
    # test the script individually
    DATA = fetch_s3_data("staywise-data-bucket/airbnb/raw_data", "AB_NYC_2019.csv")
    if DATA is not None:
        print(DATA.head())