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

def main():
    base_path = '/home/gpulab2/muhiddin/hand-gesture-recognition-mediapipe/'  # Change this to your base path
    datasets = read_datasets(base_path)

    print("Keypoint CSV data:")
    print(datasets['keypoint_csv'].head())

    print("\nKeypoint Labels:")
    print(datasets['keypoint_labels'].head())

    print("\nPoint History CSV data:")
    print(datasets['point_history_csv'].head())

    print("\nPoint History Labels:")
    print(datasets['point_history_labels'].head())

if __name__ == "__main__":
    main()
