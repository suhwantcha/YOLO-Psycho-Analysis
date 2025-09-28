import os
import json
from ultralytics import YOLO
# from IPython.display import Image, display


ROOT_DIR = '/content/drive/MyDrive/DeepLearning' 

try:
    from scoring_system import get_scores
    from case_classifier import get_analysis_label, get_script
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}. scoring_system.py와 case_classifier.py 파일이 ROOT_DIR에 있는지 확인하세요.")
    sys.exit(1) 

# 이미지 크기 설정 
IMG_W, IMG_H = 1280, 1280
MODEL_SAVE_NAME = "yolov8s-final-real" 


# --- 1. 모델 로드  ---
MODEL_PATH = os.path.join(ROOT_DIR, "detections", MODEL_SAVE_NAME, "weights", "best.pt")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ 모델 로드 완료: {os.path.basename(MODEL_PATH)}")
except Exception as e:
    model = None
    print(f"❌ 모델 로드 실패: {e}")
    
# --- 2. 스크립트 데이터 로드 ---
SCRIPTS_JSON_PATH = os.path.join(ROOT_DIR, 'scripts.json')
SCRIPT_DATA = {}
try:
    with open(SCRIPTS_JSON_PATH, 'r', encoding='utf-8') as f:
        SCRIPT_DATA = json.load(f)
except Exception as e:
    print(f"⚠️ scripts.json 파일 로드 실패: {e}. 분석 스크립트 출력이 제한됩니다.")


# --- 3. 분석 함수 정의 ---
def analyze_house_drawing(image_path):
    
    if model is None:
        print("분석을 위한 모델이 로드되지 않아 탐지를 시작할 수 없습니다.")
        return
    
    # --- 3.1 객체 탐지 (Detect) ---
    results = model(
        image_path, 
        imgsz=IMG_W, 
        conf=0.25, 
        save=True,            
        project=os.path.join(ROOT_DIR, "runs"), 
        name="analysis_results", 
        verbose=False
    )
    
    detections = []

    for r in results:
        pixel_coords = r.boxes.xyxy.tolist() # 픽셀 좌표 [x1, y1, x2, y2]
        
        for i, box in enumerate(r.boxes):
            cls = int(box.cls)
            detection_coords = pixel_coords[i] 
            
            x1, y1, x2, y2 = detection_coords
            
            detections.append({
                'class_id': cls,
                'label': model.names[cls],
                
                # 정규화 좌표
                'norm_x_center': (x1 + x2) / (2 * IMG_W),
                'norm_y_center': (y1 + y2) / (2 * IMG_H),

                # 픽셀 좌표
                'pixel_x1': x1,
                'pixel_y1': y1,
                'pixel_x2': x2,
                'pixel_y2': y2
            })

    print(f"✅ 탐지 완료: 총 {len(detections)}개의 객체 발견.")

    # --- 3.2 분석 이미지 시각화 출력 ---
    
    result_img_path = os.path.join(results[0].save_dir, os.path.basename(image_path))
    
    print("\n--- 🖼️ 분석 이미지 (BBox 포함) ---")
    if os.path.exists(result_img_path):
        display(Image(filename=result_img_path, width=500))
    else:
        print(f"❌ 시각화 이미지 저장 경로를 찾을 수 없습니다: {result_img_path}")


    # --- 3.3 심리 분석 및 결과 출력 ---
    psych_scores = get_scores(detections, IMG_W, IMG_H)
    final_label = get_analysis_label(psych_scores)
    analysis_script = get_script(final_label, SCRIPT_DATA)

    print("\n" + "="*40)
    print("           ✨ 최종 심리 분석 결과 ✨")
    print("="*40)
    print(f"✔️ 최종 심리 케이스: '{final_label}'")
    
    print("\n[ 심리 지표 점수 ]")
    for key, value in psych_scores.items():
        print(f"  - {key.capitalize():<10}: {value:.2f} 점")
    
    print("\n[ 분석 스크립트 ]")
    print(analysis_script)
    print("="*40 + "\n")
    
    return psych_scores, final_label, analysis_script

# --- 최종 실행 ---
TEST_IMAGE_PATH = os.path.join(ROOT_DIR, 'data', 'Validation', 'Images', '0001.jpg') 

print(f"\n--- 4단계: 심리 분석 실행 ---")

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"⚠️ 테스트 이미지 파일({TEST_IMAGE_PATH})을 찾을 수 없습니다. 분석을 위해 파일을 해당 경로에 업로드하거나 경로를 수정하세요.")
elif model is not None:
    analyze_house_drawing(TEST_IMAGE_PATH)