import pandas as pd
import os

def read_datasets(base_path):
    datasets = {}

    # Read keypoint classifier data
    keypoint_path = os.path.join(base_path, 'model', 'keypoint_classifier')
    datasets['keypoint_csv'] = pd.read_csv(os.path.join(keypoint_path, 'keypoint.csv'))
    datasets['keypoint_labels'] = pd.read_csv(os.path.join(keypoint_path, 'keypoint_classifier_label.csv'))
    
    # Read point history classifier data
    point_history_path = os.path.join(base_path, 'model', 'point_history_classifier')
    datasets['point_history_csv'] = pd.read_csv(os.path.join(point_history_path, 'point_history.csv'))
    datasets['point_history_labels'] = pd.read_csv(os.path.join(point_history_path, 'point_history_classifier_label.csv'))

    return datasets

def print_dataset_info(name, df, file_path):
    """Print detailed info about a dataset."""
    print(f"\n=== {name} ===")
    
    # File size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File Path: {file_path}")
    print(f"File Size: {file_size_mb:.2f} MB")
    
    # DataFrame shape
    rows, cols = df.shape
    print(f"Shape: {rows} rows, {cols} columns")
    
    # Data types
    print("Column Data Types:")
    print(df.dtypes)
    
    # First few rows
    print("First 5 Rows:")
    print(df.head())
    
    # Basic statistics (for numeric columns)
    print("Basic Statistics:")
    print(df.describe())

def main():
    base_path = '/home/gpulab2/muhiddin/hand-gesture-recognition-mediapipe/'  # Your base path
    datasets = read_datasets(base_path)

    # Print detailed info for each dataset
    print_dataset_info(
        "Keypoint CSV Data",
        datasets['keypoint_csv'],
        os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint.csv')
    )
    
    print_dataset_info(
        "Keypoint Labels",
        datasets['keypoint_labels'],
        os.path.join(base_path, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')
    )
    
    print_dataset_info(
        "Point History CSV Data",
        datasets['point_history_csv'],
        os.path.join(base_path, 'model', 'point_history_classifier', 'point_history.csv')
    )
    
    print_dataset_info(
        "Point History Labels",
        datasets['point_history_labels'],
        os.path.join(base_path, 'model', 'point_history_classifier', 'point_history_classifier_label.csv')
    )

if __name__ == "__main__":
    main()