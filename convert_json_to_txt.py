import os
import json

# 클래스 이름과 ID 매핑 [cite: 44, 45, 46, 47, 48]
LABEL_MAP = {
    "집전체": 0, "문": 1, "창문": 2, "울타리": 3, "길": 4, 
    "태양": 5, "굴뚝": 6, "지붕": 7, "연기": 8, "나무": 9, 
    "꽃": 10, "잔디": 11, "연못": 12
}

def convert_json_to_txt(json_dir, output_dir, img_w=1280, img_h=1280):
    os.makedirs(output_dir, exist_ok=True)
    
    # 임의의 JSON 파일 생성 (예시)
    sample_json = {
        "annotations": {
            "bbox": [
                {"label": "집전체", "x": 400, "y": 400, "w": 300, "h": 250},
                {"label": "문", "x": 510, "y": 550, "w": 50, "h": 80},
                {"label": "창문", "x": 450, "y": 500, "w": 40, "h": 40},
                {"label": "태양", "x": 1000, "y": 150, "w": 100, "h": 100}
            ]
        }
    }
    with open("temp_data.json", "w") as f:
        json.dump(sample_json, f)

    filename = "temp_data.json"
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(output_dir, base_name + ".txt")
    
    lines = []
    for obj in sample_json["annotations"]["bbox"]:
        label = obj["label"]
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]

        # YOLO 포맷으로 정규화 [cite: 31, 32, 33]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        class_id = LABEL_MAP[label]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    
    os.remove("temp_data.json")
    print(f"변환 완료: {filename} -> {txt_path}")

convert_json_to_txt(".", ".")