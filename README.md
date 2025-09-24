### **AI 집 그림 심리 분석 모델 (YOLO 기반)**

이 프로젝트는 딥러닝 객체 탐지 모델인 YOLO를 활용하여 사용자가 그린 **집 그림**을 분석하고, 이를 바탕으로 심리적 특성을 진단하는 AI 모델입니다. 그림 속의 다양한 객체(집, 문, 창문, 태양 등)를 탐지하고, 그 위치, 크기, 개수 등의 정보를 심리 지표 점수로 변환합니다. 최종적으로 계산된 점수를 바탕으로 사용자의 성향을 분류하고 맞춤형 심리 분석 스크립트를 제공합니다.

-----

### **프로젝트 구성**

프로젝트는 크게 세 단계로 구성되어 있습니다.

1.  **데이터 준비**: YOLO 학습을 위해 원본 JSON 파일을 TXT 형식으로 변환합니다.
2.  **모델 학습**: YOLOv8 모델을 학습하여 그림 속 객체를 정확하게 탐지합니다.
3.  **결과 분석 및 출력**: 탐지된 객체 정보를 바탕으로 심리 지표 점수를 계산하고, 심리 분석 스크립트를 생성합니다.

-----

### **파일 설명 및 실행 방법**

아래 파일들은 프로젝트의 각 단계를 수행하는 데 필요한 핵심 구성 요소입니다.

#### **1. `convert_json_to_txt.py`**

  - **설명**: YOLO 모델은 `class_id`, `x_center`, `y_center`, `width`, `height` 형식의 정규화된 좌표를 가진 `.txt` 파일을 학습 데이터로 사용합니다. 이 스크립트는 원본 데이터의 JSON 파일을 YOLO 포맷의 `.txt` 파일로 변환합니다.
  - **실행 방법**:
    1.  원본 JSON 파일들이 있는 디렉터리를 준비합니다.
    2.  스크립트 내의 `json_dir`과 `output_dir` 변수를 실제 경로로 수정합니다.
    3.  터미널에서 아래 명령어를 실행합니다.
        ```bash
        python convert_json_to_txt.py
        ```

#### **2. `data.yaml`**

  - **설명**: 이 파일은 YOLO 모델이 학습 및 검증 데이터를 올바르게 인식하도록 설정하는 파일입니다. 데이터셋의 경로, 클래스 수 (`nc`), 그리고 각 클래스 이름 (`names`)이 정의되어 있습니다.
  - **실행 방법**:
      - 이 파일은 직접 실행하는 파일이 아니며, YOLO 학습 시 `--data` 인자로 전달됩니다. 데이터셋의 실제 경로에 맞게 내용을 수정해야 합니다.

#### **3. `train_model.py` (YOLOv8 모델 학습)**

  - **설명**: YOLOv8 모델을 사용하여 집 그림 객체 탐지 모델을 학습하는 스크립트입니다. PDF에서 YOLOv5와 YOLOv8을 비교한 후 YOLOv8이 더 우수한 성능을 보여 최종 모델로 선정되었습니다.
  - **실행 방법**:
    1.  `ultralytics` 라이브러리를 설치합니다.
        ```bash
        pip install ultralytics
        ```
    2.  `train_model.py` 파일을 생성하고 아래 코드를 작성합니다.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8s.pt")  # 사전 학습된 YOLOv8s 모델 로드

        # 모델 학습
        model.train(
            data="data.yaml",           # 위에서 생성한 data.yaml 파일 지정
            epochs=150,                 # 학습 에포크 수
            imgsz=1280,                 # 이미지 크기
            batch=16,                   # 배치 크기
            project="detections",       # 프로젝트 이름
            name="yolov8s",             # 실행 이름
            pretrained=True,            # 사전 학습 모델 사용
            patience=20                 # 조기 종료(Early stopping) 기준
        )
        ```
    3.  터미널에서 아래 명령어를 실행합니다.
        ```bash
        python train_model.py
        ```

#### **4. `scoring_system.py`**

  - **설명**: 학습된 모델의 탐지 결과를 입력받아 **개방성, 활력성, 온화성, 현실 기반 안정성**이라는 네 가지 심리 지표의 점수를 계산합니다. 각 지표는 특정 객체의 속성에 따라 점수가 부여됩니다.
  - **실행 방법**:
      - 이 스크립트는 YOLO 모델의 예측 결과를 입력으로 사용합니다. 학습된 모델이 예측한 객체 정보(클래스, 위치, 크기 등)를 이 스크립트의 함수에 전달하여 점수를 계산할 수 있습니다.

#### **5. `case_classifier.py` & `scripts.json`**

  - **설명**: `scoring_system.py`에서 계산된 점수를 바탕으로 사용자의 심리 케이스를 분류하고, 미리 작성된 스크립트 파일을 참조하여 맞춤형 분석 스크립트를 출력합니다. 점수가 50점 이상이면 대문자로, 50점 미만이면 소문자로 표기하여 총 24가지 케이스로 분류합니다.
  - **실행 방법**:
      - `scripts.json` 파일에 24가지 케이스별 특징을 담은 스크립트를 미리 작성해 둡니다.
      - `case_classifier.py` 스크립트를 실행하여 `scoring_system.py`에서 얻은 점수를 입력하면, 해당 케이스에 맞는 스크립트가 출력됩니다.

-----

### **참고 자료 (References)**

[cite\_start]이 프로젝트를 위해 다음 자료들을 참고했습니다[cite: 1]:

  - [cite\_start]**AI Hub 데이터셋**: `[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSn=71399](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71399)` [cite: 1]
  - **Jolles, I. (1964). [cite\_start]A Guide to the House-Tree-Person Test.**[cite: 1]: 집 그림 심리 분석의 이론적 토대를 제공하는 서적입니다.
  - **Rubin, J. (1984). [cite\_start]The Art of Art Therapy**[cite: 1]: 그림을 통한 심리 분석에 대한 광범위한 이해를 돕기 위해 참고된 미술 치료 관련 서적입니다.

