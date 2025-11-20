import os
import json
import datetime
import tensorflow as tf
import psutil
from typing import Optional, Dict, Any


class FLResultManager:
    """Federated Learning 실험 결과를 저장하고 관리하는 클래스"""
    
    def __init__(self, execution_folder, device,  base_dir: str = "results"):
        """
        Args:
            base_dir: 결과를 저장할 기본 디렉토리 (기본값: "results")
        """
        self.base_dir = base_dir
        
        self.execution_folder = execution_folder
            
        # 디바이스 확인 (GPU/CPU)
        self.device = device
        
        # JSON 파일 경로 설정
        # 예: results/gpu/20231026_123456.json
        self.json_file_path = os.path.join(
            self.base_dir, 
            self.device, 
            f"{self.execution_folder}.json"
        )
        
        # 디렉토리 생성
        self._create_directories()
        
        # JSON 파일 초기화
        self._initialize_json_file()
        
        print(f"📁 결과 저장 폴더: {self.execution_folder}")
        print(f"📄 JSON 파일 경로: {self.json_file_path}")
        print(f"🖥️  실행 장치: {self.device.upper()}")
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        # 실행 폴더
        if not os.path.exists(self.execution_folder):
            os.makedirs(self.execution_folder)
        
        # JSON 파일 디렉토리
        json_dir = os.path.dirname(self.json_file_path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
    
    def _initialize_json_file(self):
        if not os.path.exists(self.json_file_path):
            with open(self.json_file_path, "w", encoding='utf-8') as f:
                json.dump([], f, indent=4, ensure_ascii=False)
    
    def _get_current_memory_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    # ========================================
    # 통합 저장 메서드
    # ========================================
    
    def save_round_result(self, 
                         round_number: int,
                         train_time: Optional[float] = None,
                         peak_memory: Optional[float] = None,
                         train_metrics: Optional[Dict[str, Any]] = None,
                         eval_metrics: Optional[Dict[str, Any]] = None,
                         additional_info: Optional[Dict[str, Any]] = None):
        """
        라운드별 결과 통합 저장 (모든 정보 포함)
        
        Args:
            round_number: 라운드 번호
            train_time: 학습 시간 (초)
            peak_memory: 최고 메모리 사용량 (MB)
            train_metrics: 학습 메트릭 {'loss': ..., 'accuracy': ...}
            eval_metrics: 평가 메트릭 {'f1': ..., 'precision': ..., 'recall': ...}
            additional_info: 추가 정보 (선택)
        """
        try:
            # 기존 데이터 읽기
            with open(self.json_file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ JSON 파일 로드 오류: {e}")
            data = []
        
        # 결과 구성
        result = {
            'round_number': round_number,
            'device': self.device,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 학습 시간
        if train_time is not None:
            result['train_time'] = round(train_time, 4)
        
        # 메모리 사용량
        if peak_memory is not None:
            result['peak_memory'] = round(peak_memory, 2)
        
        # 학습 메트릭
        if train_metrics:
            result['train'] = train_metrics
        
        # 평가 메트릭
        if eval_metrics:
            result['eval'] = eval_metrics
        
        # 추가 정보
        if additional_info:
            result.update(additional_info)
        
        # 새 결과 추가
        data.append(result)
        
        # 파일에 저장
        with open(self.json_file_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        # 로그 출력
        log_parts = [f"Round {round_number}"]
        if train_time is not None:
            log_parts.append(f"학습시간: {train_time:.4f}초")
        if peak_memory is not None:
            log_parts.append(f"메모리: {peak_memory:.2f}MB")
        
        print(f"✅ {' | '.join(log_parts)} → {self.json_file_path}")
    
    def save_round_time(self, round_number: int, train_time: float, peak_memory: float):
        """
        원본 save_round_time 호환 메서드
        
        Args:
            round_number: 라운드 번호
            train_time: 학습 시간 (초)
            peak_memory: 최고 메모리 사용량 (MB)
        """
        self.save_round_result(
            round_number=round_number,
            train_time=train_time,
            peak_memory=peak_memory
        )
    
    def save_train_result(self, round_number: int, loss: float, accuracy: float, 
                         train_time: Optional[float] = None):
        """
        학습 결과 저장
        
        Args:
            round_number: 라운드 번호
            loss: 학습 손실
            accuracy: 학습 정확도
            train_time: 학습 시간 (선택)
        """
        self.save_round_result(
            round_number=round_number,
            train_time=train_time,
            train_metrics={'loss': loss, 'accuracy': accuracy}
        )
    
    def save_eval_result(self, round_number: int, eval_metrics: dict):
        """
        평가 결과 저장
        
        Args:
            round_number: 라운드 번호
            eval_metrics: 평가 지표 딕셔너리 (예: {'f1': 0.85, 'precision': 0.88, 'recall': 0.82})
        """
        self.save_round_result(
            round_number=round_number,
            eval_metrics=eval_metrics
        )
    
    # ========================================
    # 유틸리티 메서드
    # ========================================
    
    def get_execution_folder(self) -> str:
        """실행 폴더 경로 반환"""
        return self.execution_folder
    
    def get_json_path(self) -> str:
        """JSON 파일 경로 반환"""
        return self.json_file_path
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device
    
    def get_all_results(self) -> list:
        """저장된 모든 결과 반환"""
        try:
            with open(self.json_file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 결과 로드 실패: {e}")
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """결과 요약 반환"""
        data = self.get_all_results()
        
        if not data:
            return {}
        
        total_rounds = len(data)
        total_train_time = sum(r.get('train_time', 0) for r in data)
        avg_memory = sum(r.get('peak_memory', 0) for r in data) / total_rounds if total_rounds > 0 else 0
        
        return {
            'total_rounds': total_rounds,
            'total_train_time': total_train_time,
            'avg_train_time_per_round': total_train_time / total_rounds if total_rounds > 0 else 0,
            'avg_memory_usage': avg_memory,
            'device': self.device,
            'execution_folder': self.execution_folder
        }


# ========================================
# 컨텍스트 매니저 (시간/메모리 자동 측정)
# ========================================

class RoundTimer:
    """라운드 학습 시간 및 메모리 자동 측정"""
    
    def __init__(self, result_manager: FLResultManager, round_number: int):
        self.result_manager = result_manager
        self.round_number = round_number
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
    
    def __enter__(self):
        """시작 시간 및 메모리 기록"""
        self.start_time = datetime.datetime.now()
        self.start_memory = self.result_manager._get_current_memory_mb()
        self.peak_memory = self.start_memory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """종료 시간 및 메모리 기록, 자동 저장"""
        end_time = datetime.datetime.now()
        end_memory = self.result_manager._get_current_memory_mb()
        
        train_time = (end_time - self.start_time).total_seconds()
        self.peak_memory = max(self.peak_memory, end_memory)
        
        # 자동 저장
        self.result_manager.save_round_time(
            round_number=self.round_number,
            train_time=train_time,
            peak_memory=self.peak_memory
        )


# ========================================
# 사용 예시
# ========================================

if __name__ == "__main__":
    # 1. ResultManager 생성
    result_manager = FLResultManager()
    
    # 2. 방법 1: 통합 저장
    result_manager.save_round_result(
        round_number=1,
        train_time=45.23,
        peak_memory=1234.56,
        train_metrics={'loss': 0.234, 'accuracy': 0.956},
        eval_metrics={'f1': 0.89, 'precision': 0.91, 'recall': 0.87}
    )
    
    # 3. 방법 2: 원본 호환 (save_round_time)
    result_manager.save_round_time(
        round_number=2,
        train_time=43.12,
        peak_memory=1250.00
    )
    
    # 4. 방법 3: 컨텍스트 매니저 (자동 측정)
    with RoundTimer(result_manager, round_number=3):
        # 학습 코드
        import time
        time.sleep(2)  # 학습 시뮬레이션
    
    # 5. 요약 정보
    summary = result_manager.get_summary()
    print(f"\n📊 실험 요약:")
    print(f"   총 라운드: {summary['total_rounds']}")
    print(f"   총 학습 시간: {summary['total_train_time']:.2f}초")
    print(f"   평균 메모리: {summary['avg_memory_usage']:.2f}MB")