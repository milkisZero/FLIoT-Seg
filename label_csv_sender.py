import pandas as pd
import socket
import os
import json
import csv

def main():
    # 1. CSV 파일 경로 입력 및 로드
    csv_path = input("CSV 파일 경로를 입력하세요 (기본값: Client/CICIDS_Splitted/client1_test.csv): ").strip()
    if csv_path == "":
        csv_path = "Client/CICIDS_Splitted/client1_test.csv"
    
    try:
        df = pd.read_csv(csv_path)
        # 헤더 컬럼명의 앞뒤 공백 제거 (예: " Label" -> "Label")
        df.columns = [col.strip() for col in df.columns]
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 2. 'Label' 컬럼 존재 여부 확인
    if 'Label' not in df.columns:
        print("CSV 파일에 'Label' 컬럼이 존재하지 않습니다.")
        print("CSV 파일의 컬럼명 목록:", df.columns.tolist())
        return

    # 3. Label 별 인덱싱 및 개수 출력
    labels = df['Label'].unique()
    print("\n존재하는 라벨 목록:")
    for i, label in enumerate(labels):
        count = (df['Label'] == label).sum()
        print(f"{i}: {label} ({count}개)")
    
    # 4. 다중 인덱스 선택 (콤마(,)로 구분) - 아무것도 입력하면 전체 라벨 선택
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
    
    # 5. 각 선택된 라벨마다 추출할 데이터 개수를 입력받고 데이터 결합
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
    
    # 6. 결과 CSV 파일 저장 (여러 라벨이 선택된 경우 이름에 모두 포함)
    joined_labels = "_".join([str(label) for label in selected_labels])
    output_filename = f"selected_labels_{joined_labels}.csv"
    try:
        result_df.to_csv(output_filename, index=False)
        print(f"\n'{output_filename}' 파일이 생성되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류가 발생했습니다: {e}")
        return

    # 7. 전송할 대상 정보 입력 (localhost 여부 먼저 확인)
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

    # 8. 각 대상에 대해 TCP 소켓을 이용해 CSV 파일의 각 행을 한 줄씩 전송 (JSON 메시지 형식)
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
                    "event": "FILE_END",
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