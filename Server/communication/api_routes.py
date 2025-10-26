"""
일반 API 엔드포인트
"""
from flask import Blueprint, jsonify, render_template
import json

def create_api_routes(fl_server):
    """API Blueprint 생성"""
    api_bp = Blueprint('api', __name__)
    
    @api_bp.route('/')
    def dashboard():
        """대시보드"""
        return render_template('dashboard.html')
    
    @api_bp.route('/stats')
    def stats():
        """학습 통계"""
        return json.dumps(fl_server.global_model.get_stats())
    
    @api_bp.route('/health')
    def health_check():
        """헬스 체크"""
        return jsonify({"status": "ok"}), 200
    
    @api_bp.route('/status')
    def status():
        """서버 상태"""
        return jsonify({
            "current_round": fl_server.protocol_handler.current_round,
            "ready_clients": len(fl_server.protocol_handler.ready_client_sids),
            "max_rounds": fl_server.config.MAx_NUM_ROUNDS
        })
    
    return api_bp