import os
import json
import glob
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_metrics_from_folder(folder_path):
    """
    지정된 폴더 내의 모든 JSON 파일을 읽어서 round_number에 따른
    train_time과 peak_memory를 그래프로 그립니다.
    
    Args:
        folder_path (str): JSON 파일이 있는 폴더 경로
    """
    # 폴더 내의 모든 JSON 파일 찾기
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"오류: {folder_path}에 JSON 파일이 없습니다.")
        return
    
    # 그래프를 위한 설정
    plt.figure(figsize=(12, 10))
    
    # 첫 번째 서브플롯: Round Number vs Train Time
    plt.subplot(2, 1, 1)
    
    # 클라이언트별로 다른 색상 및 마커 사용
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    
    # 각 JSON 파일에 대해 처리
    for i, json_file in enumerate(sorted(json_files)):
        client_name = os.path.basename(json_file).split('.')[0]
        
        # JSON 파일 읽기
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # round_number와 train_time 추출
        round_numbers = [entry["round_number"] for entry in data]
        train_times = [entry["train_time"] for entry in data]
        
        # Train Time 그래프에 데이터 추가
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        plt.plot(round_numbers, train_times, 
                 marker=markers[marker_idx], 
                 linestyle='-', 
                 label=client_name,
                 color=colors[color_idx])
    
    plt.title('Round Number vs Training Time')
    plt.xlabel('Round Number')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 두 번째 서브플롯: Round Number vs Peak Memory
    plt.subplot(2, 1, 2)
    
    # 각 JSON 파일에 대해 처리
    for i, json_file in enumerate(sorted(json_files)):
        client_name = os.path.basename(json_file).split('.')[0]
        
        # JSON 파일 읽기
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # round_number와 peak_memory 추출
        round_numbers = [entry["round_number"] for entry in data]
        peak_memories = [entry["peak_memory"] for entry in data]
        
        # Peak Memory 그래프에 데이터 추가
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        plt.plot(round_numbers, peak_memories, 
                 marker=markers[marker_idx], 
                 linestyle='-', 
                 label=client_name,
                 color=colors[color_idx])
    
    plt.title('Round Number vs Peak Memory')
    plt.xlabel('Round Number')
    plt.ylabel('Peak Memory (MB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 그래프 간격 조정 및 저장
    plt.tight_layout()
    
    # 결과 폴더 생성
    output_dir = os.path.join(os.path.dirname(folder_path), "graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 그래프 저장
    output_file = os.path.join(output_dir, f"{os.path.basename(folder_path)}_metrics.png")
    plt.savefig(output_file, dpi=300)
    print(f"그래프가 저장되었습니다: {output_file}")
    
    # 그래프 표시
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 파일에서 train_time과 peak_memory를 그래프로 그립니다.")
    parser.add_argument("folder", help="JSON 파일이 있는 폴더 경로")
    args = parser.parse_args()
    
    plot_metrics_from_folder(args.folder)