import pandas as pd
import socket
import os
import json
import csv
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
import argparse
from dotenv import load_dotenv

load_dotenv('.env')

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
    
    if "Label" not in input_df.columns:
        raise ValueError("CSV 파일에 'Label' 컬럼이 존재하지 않습니다.")
    
    available_features = [col for col in numerical_features if col in input_df.columns]
    
    # 원본 레이블 복원용 변수 저장 (수정 전 그대로 보존)
    original_labels = input_df["Label"].copy()
    
    # 기존 'Label' 컬럼 제거 후, 필요한 전처리 진행 (단, 이진 변환 대신 원본 라벨 유지)
    # 만약 전처리 목적이 단순 스케일링이면 그대로 진행
    df = input_df[available_features].copy()
    
    # 수치형 특성 정규화
    for col in available_features:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    
    # 전처리된 데이터와 함께 원본 레이블을 그대로 추가 (원-핫 인코딩은 서버쪽에서 진행)
    df["Label"] = original_labels
    
    # binary_labels 대신 다중 클래스 처리를 위한 방식으로 변경할 수 있음
    return df, original_labels

def list_csv_files(directory):
    """지정된 디렉토리에 있는 모든 CSV 파일 목록을 반환"""
    pattern = os.path.join(directory, "*.csv")
    return glob.glob(pattern)

def main():
    # 명령행 인자 처리 (자동 입력 모드)
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="자동 입력 모드 활성화")
    args = parser.parse_args()
    auto = args.auto

    # 1. CSV 파일 선택
    cicids_dir = "Client/CICIDS"
    
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
    
    if auto:
        use_all = "y"
        print("\n[자동 입력] 모든 CSV 파일을 합쳐서 사용합니다.")
    else:
        use_all = input("\n모든 CSV 파일을 합쳐서 사용하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    
    if use_all == 'y':
        all_dfs = []
        for csv_file in csv_files:
            try:
                print(f"{os.path.basename(csv_file)} 파일 로딩 중...")
                temp_df = pd.read_csv(csv_file)
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
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"전체 {len(csv_files)}개 파일 중 {len(all_dfs)}개 파일을 결합했습니다.")
    else:
        if auto:
            print("[자동 입력] 자동 모드에서는 전체 파일을 사용합니다.")
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        else:
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
                df.columns = df.columns.str.strip()
                print(f"{os.path.basename(csv_path)} 파일을 로드했습니다.")
            except Exception as e:
                print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
                return
    
    if 'Label' not in df.columns:
        print("CSV 파일에 'Label' 컬럼이 존재하지 않습니다.")
        print("CSV 파일의 컬럼명 목록:", df.columns.tolist())
        return

    if auto:
        apply_preprocessing = "y"
        print("[자동 입력] 데이터 전처리를 적용합니다.")
    else:
        apply_preprocessing = input("데이터 전처리를 적용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
    
    if apply_preprocessing == "" or apply_preprocessing == "y":
        try:
            print("\n데이터 전처리를 시작합니다...")
            X, original_labels = preprocess_data(df)
            processed_df = pd.DataFrame(X)
            # 원래 라벨을 그대로 사용
            processed_df["Label"] = original_labels  # 원래 라벨을 그대로 사용
            df = processed_df
            print("데이터 전처리가 완료되었습니다.")
        except Exception as e:
            print(f"데이터 전처리 중 오류가 발생했습니다: {e}")
            print("전처리 없이 원본 데이터를 사용합니다.")
    else:
        print("전처리 없이 원본 데이터를 사용합니다.")
        df["Label"] = df["Label"].apply(lambda x: "nomaly" if str(x).strip().upper() == "BENIGN" else "anomaly")

    labels = df['Label'].unique()
    print("\n존재하는 라벨 목록:")
    for i, label in enumerate(labels):
        count = (df['Label'] == label).sum()
        print(f"{i}: {label} ({count}개)")
    
    if auto:
        selected_indexes_input = ""
        print("[자동 입력] 전체 라벨을 선택합니다.")
    else:
        selected_indexes_input = input("\n추출할 라벨의 인덱스를 콤마(,)로 구분하여 선택하세요 (아무것도 입력하면 전체 라벨 선택): ").strip()
    
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
    
    result_dfs = []
    for label in selected_labels:
        if auto:
            num_samples = 1000
            print(f"[자동 입력] '{label}' 라벨에서 추출할 데이터 개수를 {num_samples}로 설정합니다.")
        else:
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
    joined_labels = "_".join([str(label) for label in selected_labels])
    output_filename = f"selected_labels_{joined_labels}.csv"
    try:
        result_df.to_csv(output_filename, index=False)
        print(f"\n'{output_filename}' 파일이 생성되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류가 발생했습니다: {e}")
        return

    target_destinations = []  # (ip, port) 튜플의 리스트
    if auto:
        use_localhost = "y"
        print("[자동 입력] 전송할 대상은 localhost 입니다.")
    else:
        use_localhost = input("전송할 대상이 localhost 입니까? (y/n): ").strip().lower()
    
    if use_localhost == 'y':
        if auto:
            client_val = os.environ.get("CLIENT")
            try:
                num_clients = int(client_val)
            except ValueError:
                num_clients = 1
            print(f"[자동 입력] 로컬호스트 대상 클라이언트 수를 {num_clients}로 설정합니다.")
        else:
            try:
                num_clients = int(input("로컬호스트 대상 클라이언트 수를 입력하세요: ").strip())
            except ValueError:
                print("올바른 클라이언트 수가 입력되지 않았습니다.")
                return
        # 각 클라이언트에 대해 포트를 4000 + 클라이언트 번호로 할당
        for client_idx in range(1, num_clients+1):
            port = 4000 + client_idx
            target_destinations.append(("localhost", port))
    else:
        if auto:
            print("[자동 입력] 비로컬호스트 전송 대상은 자동 입력 모드에서 지원되지 않습니다.")
            return
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
    
    # 각 대상에 대해 TCP 소켓을 이용해 CSV 파일의 각 행(한 줄씩 전송)
    for target_ip, target_port in target_destinations:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"\n{target_ip}:{target_port} 로 연결 시도 중...")
            sock.connect((target_ip, target_port))
            print("연결 성공, 파일 전송 시작합니다.")
            
            with open(output_filename, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # 헤더 스킵
                for row in reader:
                    if not row:
                        continue
                    label = row[-1]
                    feature_strings = row[:-1]
                    features = []
                    for item in feature_strings:
                        try:
                            features.append(float(item))
                        except ValueError:
                            features.append(item)
                    
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
                    message_length = len(message_bytes)
                    length_bytes = message_length.to_bytes(4, byteorder="big")
                    sock.sendall(length_bytes + message_bytes)
            
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