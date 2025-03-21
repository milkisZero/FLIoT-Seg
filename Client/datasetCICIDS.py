#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

def gen_train_valid_data(num_classes, benign_only=False):
    # Read the client_1 train and test files.
    training_df = pd.read_csv("CICIDS_Splitted/client_train.csv")
    testing_df = pd.read_csv("CICIDS_Splitted/client_test.csv")

    # Strip whitespace from column names to ensure consistency.
    training_df.columns = training_df.columns.str.strip()
    testing_df.columns = testing_df.columns.str.strip()

    # The union of numerical features across all files.
    numerical_features = [
        'ACK Flag Count', 'Active Max', 'Active Min', 'Active Std', 'Average Packet Size',
        'Avg Bwd Segment Size', 'Avg Fwd Segment Size', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
        'Bwd Header Length', 'Bwd IAT Max', 'Bwd IAT Mean', 'Bwd IAT Min', 'Bwd IAT Std',
        'Bwd PSH Flags', 'Bwd Packet Length Mean', 'Bwd Packet Length Min', 'Bwd Packet Length Std',
        'Bwd Packets/s', 'Bwd URG Flags', 'CWE Flag Count', 'Destination Port', 'Down/Up Ratio',
        'ECE Flag Count', 'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min',
        'Flow IAT Std', 'Flow Packets/s', 'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'Fwd Header Length',
        'Fwd Header Length.1', 'Fwd IAT Max', 'Fwd IAT Mean', 'Fwd IAT Min', 'Fwd IAT Std',
        'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Min', 'Fwd Packet Length Std',
        'Fwd URG Flags', 'Idle Max', 'Idle Min', 'Idle Std', 'Init_Win_bytes_backward', 'Max Packet Length',
        'Min Packet Length', 'PSH Flag Count', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'RST Flag Count', 'SYN Flag Count', 'Subflow Bwd Bytes',
        'Subflow Bwd Packets', 'Subflow Fwd Bytes', 'Total Backward Packets', 'Total Fwd Packets',
        'Total Length of Bwd Packets', 'URG Flag Count', 'act_data_pkt_fwd', 'min_seg_size_forward',
        'Active Mean', 'Bwd Avg Bulk Rate', 'Bwd IAT Total', 'Bwd Packet Length Max', 'FIN Flag Count',
        'Flow Bytes/s', 'Fwd Avg Bytes/Bulk', 'Fwd IAT Total', 'Fwd PSH Flags', 'Fwd Packets/s',
        'Idle Mean', 'Init_Win_bytes_forward', 'Subflow Fwd Packets', 'Total Length of Fwd Packets'
    ]

    # Ensure the label column ("Label") is kept.
    if "Label" not in training_df.columns:
        raise ValueError("The training CSV file does not have a 'Label' column.")
    if "Label" not in testing_df.columns:
        raise ValueError("The testing CSV file does not have a 'Label' column.")

    # Keep only the numerical features (if they exist) plus the "Label" column.
    train_cols = [col for col in numerical_features if col in training_df.columns]
    test_cols = [col for col in numerical_features if col in testing_df.columns]

    # It is assumed that both train and test contain the same numerical features.
    training_df = training_df[train_cols + ["Label"]]
    testing_df = testing_df[test_cols + ["Label"]]

    # Define a function to transform labels.
    # We will create a new column "Class": if Label equals "BENIGN" (case insensitive) then Class = "BENIGN",
    # otherwise, Class = "Attack".
    def label_transform(row):
        return "BENIGN" if str(row["Label"]).strip().upper() == "BENIGN" else "Attack"

    # Apply the transformation.
    training_df["Class"] = training_df.apply(label_transform, axis=1)
    testing_df["Class"] = testing_df.apply(label_transform, axis=1)
    
    # 원래 Label 보존 (이후 JSON 저장 시 활용)
    orig_y_train = training_df["Label"].values.copy()
    orig_y_test = testing_df["Label"].values.copy()

    # Drop the original Label column.
    training_df.drop("Label", axis=1, inplace=True)
    testing_df.drop("Label", axis=1, inplace=True)

    # Apply MinMax scaling for each numerical column.
    for col in train_cols:
        # Replace infinite values with NaN
        training_df[col] = training_df[col].replace([np.inf, -np.inf], np.nan)
        testing_df[col] = testing_df[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaNs with the median of the column (you can also choose 0 or another strategy)
        training_df[col].fillna(training_df[col].median(), inplace=True)
        testing_df[col].fillna(testing_df[col].median(), inplace=True)
        # Create a new scaler instance for each column.
        scaler = MinMaxScaler()
        training_df[col] = scaler.fit_transform(training_df[col].values.reshape(-1, 1))
        testing_df[col] = scaler.transform(testing_df[col].values.reshape(-1, 1))

    # Now separate features and labels.
    x, y = training_df, training_df.pop("Class").values
    X_train = x.values
    x_test, y_test = testing_df, testing_df.pop("Class").values
    X_test = x_test.values
    # Create binary labels:
    # We assume that "BENIGN" corresponds to 0 and any other label ("Attack") to 1.
    y_train = np.ones(len(y), np.int8)
    y_train[np.where(y == "BENIGN")] = 0

    y_test_binary = np.ones(len(y_test), np.int8)
    y_test_binary[np.where(y_test == "BENIGN")] = 0

    # 추가: 라벨을 one-hot 인코딩합니다.
    # 모델의 출력이 3개 노드이므로 num_classes=3으로 설정합니다.
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test_binary = to_categorical(y_test_binary, num_classes=num_classes)
    
    # 모델 입력 형태에 맞게 데이터 재구조화: 각 샘플을 (특성 수, 1)로 reshape
    X_train = X_train.reshape((-1, len(train_cols), 1))
    X_test = X_test.reshape((-1, len(train_cols), 1))
    
    if benign_only:
        indices = np.where(y_train[:, 0] == 1)[0]  # BENIGN 데이터는 one-hot에서 첫번째 인덱스가 1
        return X_train[indices], y_train[indices], X_test, y_test_binary, orig_y_train[indices], orig_y_test
    else:
        return X_train, y_train, X_test, y_test_binary, orig_y_train, orig_y_test