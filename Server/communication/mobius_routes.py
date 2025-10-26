from flask import Blueprint, request, jsonify

def create_mobius_routes(fl_server):
    mobius_bp = Blueprint('mobius', __name__)
       
    # aeWatcher 알림 엔드포인트
    @mobius_bp.route('/aeWatcher', methods=['POST'])
    def aeWatcher():
        try:
            data = request.json
            print("Received AE:")

            if "m2m:sgn" in data:
                # rn인지 ri인지 선택해야 함
                content = data["m2m:sgn"]["nev"]["rep"]["m2m:ae"]["rn"]
                print(f"New AE detected(rn): {content}")
                fl_server.mobius_handler.create_aeSub(content)
                                
            return jsonify({"status": "received"}), 200

        except Exception as e:
            print(f"Error processing notification: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # aeSub 알림 엔드포인트
    @mobius_bp.route('/aeSub', methods=['POST'])
    def aeSub():
        try:
            data = request.json
            print("Received Container:")

            if "m2m:sgn" in data:
                # rn인지 ri인지 선택해야 함
                content = data["m2m:sgn"]["nev"]["rep"]["m2m:cnt"]["rn"]
                print(f"New CNT detected(rn): {content}")
                url = data["m2m:sgn"]['sur']
                url = url[:url.rfind('/')+1]
                
                if content[len(content)-1] == 'S':
                    fl_server.mobius_handler.pub_cnt_list.append(content)
                elif content[len(content)-1] == 'C':
                    fl_server.mobius_handler.create_cntSub(url + content, content)
                
            return jsonify({"status": "received"}), 200

        except Exception as e:
            print(f"Error processing notification: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # 데이터 엔드포인트
    @mobius_bp.route('/notify', methods=['POST'])
    def notify():
        try:
            # Mobius에서 전송된 알림 데이터
            data = request.json
            print("Received Notification:")

            # 알림 데이터에서 콘텐츠 추출
            if "m2m:sgn" in data:
                content = data["m2m:sgn"]["nev"]["rep"]["m2m:cin"]["con"]
                #print("Updated Content: ", content)   
                
                url = data["m2m:sgn"]['sur']
                client_id = url.split('/')[2].replace("FromC", "")
                fl_server.protocol_handler.on_message(content, client_id)

            return jsonify({"status": "received"}), 200

        except Exception as e:
            print(f"Error processing notification: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
        
    return mobius_bp

    