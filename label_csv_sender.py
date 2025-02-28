import pandas as pd
import socket
import os
import json
import csv
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_df):
    """CICIDS 데이터셋을 전처리하는 함수"""
    # 컬럼명 공백 제거
    input_df.columns = input_df.columns.str.strip()
    
    # 수치형 특성 목록
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
    
    # 'Label' 컬럼 존재 확인
    if "Label" not in input_df.columns:
        raise ValueError("CSV 파일에 'Label' 컬럼이 존재하지 않습니다.")
    
    # 존재하는 수치형 특성만 선택
    available_features = [col for col in numerical_features if col in input_df.columns]
    
    # 필요한 컬럼만 선택 (수치형 특성 + Label)
    df = input_df[available_features + ["Label"]].copy()
    
    # 레이블 변환 함수: BENIGN -> nomaly, 나머지 -> anomaly
    def label_transform(row):
        return "nomaly" if str(row["Label"]).strip().upper() == "BENIGN" else "anomaly"
    
    # 새로운 Class 컬럼 추가
    df["Class"] = df.apply(label_transform, axis=1)
    
    # 원본 Label 컬럼 저장 (나중에 사용)
    original_labels = df["Label"].copy()
    
    # 원본 Label 컬럼 제거
    df.drop("Label", axis=1, inplace=True)
    
    # 수치형 특성 정규화 (MinMax 스케일링)
    for col in available_features:
        # 무한값을 NaN으로 변환
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # NaN 값을 중앙값으로 채움
        df[col].fillna(df[col].median(), inplace=True)
        # MinMax 스케일링 적용
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    
    # 이진 레이블 생성 (nomaly=0, anomaly=1)
    binary_labels = np.ones(len(df), np.int8)
    binary_labels[np.where(df["Class"] == "nomaly")] = 0
    
    # 전처리된 데이터와 함께 원본 레이블도 반환
    return df[available_features], df["Class"], binary_labels, original_labels

def list_csv_files(directory):
    """지정된 디렉토리에 있는 모든 CSV 파일 목록을 반환"""
    pattern = os.path.join(directory, "*.csv")
    return glob.glob(pattern)

def main():
    # 1. CSV 파일 선택
    cicids_dir = "Client/CICIDS"
    
    # Client/CICIDS 디렉토리가 존재하는지 확인
    if not os.path.exists(cicids_dir):
        print(f"'{cicids_dir}' 디렉토리가 존재하지 않습니다.")
        cicids_dir = input("CICIDS 데이터셋이 있는 디렉토리 경로를 입력하세요: ").strip()
        if not os.path.exists(cicids_dir):
            print(f"'{cicids_dir}' 디렉토리가 존재하지 않습니다.")
            return
    
    # 디렉토리에서 모든 CSV 파일 목록 가져오기
    csv_files = list_csv_files(cicids_dir)
    
    if not csv_files:
        print(f"'{cicids_dir}' 디렉토리에 CSV 파일이 없습니다.")
        return
    
    print("\n사용 가능한 CSV 파일 목록:")
    for i, csv_file in enumerate(csv_files):
        print(f"{i+1}. {os.path.basename(csv_file)}")
    
    # 사용자가 선택하거나 모든 파일 사용
    use_all = input("\n모든 CSV 파일을 합쳐서 사용하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    
    if use_all == 'y':
        # 모든 CSV 파일 합치기
        all_dfs = []
        for csv_file in csv_files:
            try:
                print(f"{os.path.basename(csv_file)} 파일 로딩 중...")
                temp_df = pd.read_csv(csv_file)
                # 즉시 컬럼명의 공백 제거
                temp_df.columns = temp_df.columns.str.strip()
                if 'Label' not in temp_df.columns:
                    print(f"경고: {os.path.basename(csv_file)}에 'Label' 컬럼이 없습니다. 이 파일은 건너뜁니다.")
                    continue
                all_dfs.append(temp_df)
            except Exception as e:
                print(f"{os.path.basename(csv_file)} 파일 로딩 중 오류 발생: {e}")
        
        if not all_dfs:
            print("유효한 CSV 파일이 없습니다.")
            return
        
        # 모든 데이터프레임 결합
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"전체 {len(csv_files)}개 파일 중 {len(all_dfs)}개 파일을 결합했습니다.")
    else:
        # 개별 파일 선택
        try:
            file_index = int(input(f"\n사용할 CSV 파일 번호를 선택하세요 (1-{len(csv_files)}): ").strip())
            if file_index < 1 or file_index > len(csv_files):
                print("잘못된 파일 번호입니다.")
                return
            csv_path = csv_files[file_index-1]
        except ValueError:
            print("올바른 숫자를 입력하세요.")
            return
        
        try:
            df = pd.read_csv(csv_path)
            # 즉시 컬럼명의 공백 제거
            df.columns = df.columns.str.strip()
            print(f"{os.path.basename(csv_path)} 파일을 로드했습니다.")
        except Exception as e:
            print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
            return
    
    # 2. 'Label' 컬럼 존재 여부 확인 (컬럼명 공백 제거 후)
    if 'Label' not in df.columns:
        print("CSV 파일에 'Label' 컬럼이 존재하지 않습니다.")
        print("CSV 파일의 컬럼명 목록:", df.columns.tolist())
        return

    # 3. 데이터 전처리 적용 여부 선택
    apply_preprocessing = input("데이터 전처리를 적용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
    if apply_preprocessing == "" or apply_preprocessing == "y":
        try:
            print("\n데이터 전처리를 시작합니다...")
            X, class_labels, binary_labels, original_labels = preprocess_data(df)
            # 전처리된 데이터를 원본 데이터프레임에 다시 병합
            processed_df = pd.DataFrame(X)
            # BENIGN은 nomaly로, 나머지는 anomaly로 변환
            nomaly_anomaly_labels = ["nomaly" if str(label).strip().upper() == "BENIGN" else "anomaly" for label in original_labels]
            processed_df["Label"] = nomaly_anomaly_labels  # 변환된 레이블로 설정
            df = processed_df
            print("데이터 전처리가 완료되었습니다.")
        except Exception as e:
            print(f"데이터 전처리 중 오류가 발생했습니다: {e}")
            print("전처리 없이 원본 데이터를 사용합니다.")
    else:
        print("전처리 없이 원본 데이터를 사용합니다.")
        # 원본 데이터에도 레이블 변환 적용
        df["Label"] = df["Label"].apply(lambda x: "nomaly" if str(x).strip().upper() == "BENIGN" else "anomaly")

    # 4. Label 별 인덱싱 및 개수 출력
    labels = df['Label'].unique()
    print("\n존재하는 라벨 목록:")
    for i, label in enumerate(labels):
        count = (df['Label'] == label).sum()
        print(f"{i}: {label} ({count}개)")
    
    # 5. 다중 인덱스 선택 (콤마(,)로 구분) - 아무것도 입력하면 전체 라벨 선택
    selected_indexes_input = input(
        "\n추출할 라벨의 인덱스를 콤마(,)로 구분하여 선택하세요 (아무것도 입력하면 전체 라벨 선택): "
    ).strip()
    if not selected_indexes_input:
        selected_indexes = list(range(len(labels)))
    else:
        try:
            selected_indexes = [int(x.strip()) for x in selected_indexes_input.split(",") if x.strip() != ""]
            if any(idx < 0 or idx >= len(labels) for idx in selected_indexes):
                print("입력된 인덱스 중 범위를 벗어난 값이 있습니다.")
                return
        except ValueError:
            print("올바른 인덱스 숫자가 입력되지 않았습니다.")
            return
    selected_labels = [labels[idx] for idx in selected_indexes]
    
    # 6. 각 선택된 라벨마다 추출할 데이터 개수를 입력받고 데이터 결합
    result_dfs = []
    for label in selected_labels:
        try:
            num_samples = int(input(f"'{label}' 라벨에서 추출할 데이터 개수를 입력하세요: "))
        except ValueError:
            print("올바른 숫자가 입력되지 않았습니다.")
            return
        filtered_df = df[df['Label'] == label]
        total_count = len(filtered_df)
        if total_count < num_samples:
            print(f"{label} 라벨의 데이터가 {total_count}개밖에 없으므로 전체 데이터를 사용합니다.")
            num_samples = total_count
        sample_df = filtered_df.head(num_samples)
        result_dfs.append(sample_df)
    
    result_df = pd.concat(result_dfs, ignore_index=True)
    
    # 7. 결과 CSV 파일 저장 (여러 라벨이 선택된 경우 이름에 모두 포함)
    joined_labels = "_".join([str(label) for label in selected_labels])
    output_filename = f"selected_labels_{joined_labels}.csv"
    try:
        result_df.to_csv(output_filename, index=False)
        print(f"\n'{output_filename}' 파일이 생성되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류가 발생했습니다: {e}")
        return

    # 8. 전송할 대상 정보 입력 (localhost 여부 먼저 확인)
    target_destinations = []  # (ip, port) 튜플의 리스트
    use_localhost = input("전송할 대상이 localhost 입니까? (y/n): ").strip().lower()
    if use_localhost == 'y':
        try:
            num_clients = int(input("로컬호스트 대상 클라이언트 수를 입력하세요: ").strip())
        except ValueError:
            print("올바른 클라이언트 수가 입력되지 않았습니다.")
            return
        # 클라이언트 수만큼 포트를 4000+클라이언트 번호로 할당
        for client_idx in range(1, num_clients+1):
            port = 4000 + client_idx
            target_destinations.append(("localhost", port))
    else:
        try:
            num_destinations = int(input("전송할 대상 IP의 개수를 입력하세요: ").strip())
        except ValueError:
            print("올바른 숫자가 입력되지 않았습니다.")
            return
        for i in range(num_destinations):
            ip = input(f"\n대상 {i+1}의 IP 주소를 입력하세요: ").strip()
            if ip == "":
                print("IP 주소가 입력되지 않았습니다.")
                return
            try:
                port = int(input(f"대상 {ip}의 포트 번호를 입력하세요: ").strip())
            except ValueError:
                print("올바른 포트 번호가 입력되지 않았습니다.")
                return
            target_destinations.append((ip, port))

    # 9. 각 대상에 대해 TCP 소켓을 이용해 CSV 파일의 각 행을 한 줄씩 전송 (JSON 메시지 형식)
    for target_ip, target_port in target_destinations:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"\n{target_ip}:{target_port} 로 연결 시도 중...")
            sock.connect((target_ip, target_port))
            print("연결 성공, 파일 전송 시작합니다.")
            
            # csv.reader를 활용해 텍스트 모드로 파일을 열고, 첫 번째 헤더 행은 건너뜁니다.
            with open(output_filename, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # 헤더 스킵
                for row in reader:
                    if not row:
                        continue
                    # 마지막 컬럼은 'Label'로 가정, 나머지는 feature 값으로 처리합니다.
                    label = row[-1]
                    feature_strings = row[:-1]
                    features = []
                    for item in feature_strings:
                        try:
                            features.append(float(item))
                        except ValueError:
                            features.append(item)
                    
                    # 메시지 구성: 내부 message에 event와 payload를 포함하고, 외부 header는 "OPERATE"
                    message_payload = {
                        "event": "classify_packet",
                        "payload": {
                            "packet": features,
                            "true_label": label
                        }
                    }
                    full_message = {
                        "header": "OPERATE",
                        "message": message_payload
                    }
                    message_json = json.dumps(full_message)
                    message_bytes = message_json.encode("utf-8")
                    # 4바이트(빅엔디안)로 메시지 길이 전송 후 메시지 전송
                    message_length = len(message_bytes)
                    length_bytes = message_length.to_bytes(4, byteorder="big")
                    sock.sendall(length_bytes + message_bytes)
            
            # 전송 완료 후, 종료 메시지 전송 (옵션)
            termination_message = {
                "header": "OPERATE",
                "message": {
                    "event": "file_end",
                    "payload": {"message": "EOF"}
                }
            }
            end_json = json.dumps(termination_message).encode("utf-8")
            end_length = len(end_json)
            sock.sendall(end_length.to_bytes(4, byteorder="big") + end_json)
            
            print(f"{target_ip}:{target_port} 에 파일 전송이 완료되었습니다.")
        except Exception as e:
            print(f"{target_ip}:{target_port} 로 전송 중 오류 발생: {e}")
        finally:
            sock.close()

if __name__ == "__main__":
    main()
