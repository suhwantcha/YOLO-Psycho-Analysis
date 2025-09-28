model = YOLO("yolov8s.pt")

# 모델 학습 
model.train(
    data="data.yaml",           # 설정 파일
    epochs=50,                  # 학습 에포크 수 [cite: 161]
    imgsz=640,                  # 이미지 크기 (속도 최적화) [cite: 162, 173]
    batch=32,                   # 배치 크기 (학습 효율 개선) [cite: 163, 174]
    project="detections",       # 저장될 프로젝트 폴더명 [cite: 165]
    name="yolov8s-final",       # 실행 및 가중치 저장 이름 [cite: 166]
    pretrained=True,            # 사전 학습 모델 사용 [cite: 167, 175]
    patience=10,                # 조기 종료(Early stopping) 기준 [cite: 168, 176]
    workers=4                   # DataLoader 병렬 처리 [cite: 170, 177]
)

# 학습 완료 후, 가중치 파일은 'detections/yolov8s-final/weights/best.pt'에 저장됨