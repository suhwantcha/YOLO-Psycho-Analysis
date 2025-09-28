import numpy as np
CLASS_NAMES = ["집전체", "문", "창문", "울타리", "길", "태양", "굴뚝", "지붕", "연기", "나무", "꽃", "잔디", "연못"]

def calculate_openness(detections, img_w, img_h):
    # [cite_start]개방성 점수 계산 [cite: 217]
    score = 30 # 기본 점수 [cite: 220]
    
    # 객체 필터링 및 데이터 준비
    house_boxes = [d for d in detections if d['label'] == '집전체']
    door_boxes = [d for d in detections if d['label'] == '문']
    window_count = len([d for d in detections if d['label'] == '창문'])
    fence_boxes = [d for d in detections if d['label'] == '울타리']
    path_exists = any(d['label'] == '길' for d in detections)
    
    if not house_boxes: return 50

    # 가장 큰 집을 기준으로 계산
    main_house = max(house_boxes, key=lambda d: (d['pixel_x2'] - d['pixel_x1']) * (d['pixel_y2'] - d['pixel_y1']))
    house_area = (main_house['pixel_x2'] - main_house['pixel_x1']) * (main_house['pixel_y2'] - main_house['pixel_y1'])

    # [cite_start]1. 문 점수 (최대 50점) [cite: 218, 223, 226, 228]
    door_area_sum = sum((d['pixel_x2'] - d['pixel_x1']) * (d['pixel_y2'] - d['pixel_y1']) for d in door_boxes)
    if house_area > 0:
        door_ratio = door_area_sum / house_area
        if door_ratio < 0.02:
            score += max(-20, (door_ratio / 0.02) * 20 - 20) # -20점까지 감소
        elif door_ratio > 0.04:
            score += min(20, (door_ratio - 0.04) / 0.01 * 5) # +20점까지 증가
    
    # [cite_start]2. 창문 점수 (최대 30점) [cite: 219, 221, 227, 229]
    score += window_count * 10
    score = min(score, 80) # 문+창문 합산 점수 제한

    # [cite_start]3. 울타리 점수 (최대 -20점) [cite: 233, 234, 235, 236]
    if fence_boxes:
        score -= 10
        main_fence = max(fence_boxes, key=lambda d: d['pixel_y2'] - d['pixel_y1'])
        fence_height = main_fence['pixel_y2'] - main_fence['pixel_y1']
        house_height = main_house['pixel_y2'] - main_house['pixel_y1']
        if house_height > 0 and fence_height / house_height >= 0.5:
            score -= 10
    
    # [cite_start]4. 길 점수 [cite: 237, 238]
    if path_exists:
        score += 20
        
    return max(0, min(100, score))

def calculate_vitality(detections, img_w, img_h):
    # [cite_start]활력성 점수 계산 [cite: 240]
    score = 0
    sun_boxes = [d for d in detections if d['label'] == '태양']
    natural_objects = [d for d in detections if d['label'] in ['나무', '꽃', '잔디', '연못']]
    
    # [cite_start]1. 태양 위치 점수 (최대 50점) [cite: 242, 247, 250, 256]
    if sun_boxes:
        sun = sun_boxes[0]
        cx, cy = sun['pixel_x1'] + (sun['pixel_x2'] - sun['pixel_x1']) / 2, sun['pixel_y1'] + (sun['pixel_y2'] - sun['pixel_y1']) / 2
        
        # 중심성 점수: 중앙에서 멀수록 감점
        center_dist = np.sqrt((cx - img_w/2)**2 + (cy - img_h/2)**2)
        center_score = 50 * (1 - center_dist / (np.sqrt(img_w**2 + img_h**2) / 2))
        
        # [cite_start]높이 점수: 상단(y=0)에 가까울수록 가점 [cite: 251, 252]
        height_score = 50 * (1 - cy / img_h)
        
        sun_score = (center_score + height_score) / 2
        score += min(50, sun_score)
        
    # [cite_start]2. 자연물 개수 점수 (최대 35점) [cite: 243, 245]
    score += min(35, len(natural_objects) * 5)
    
    # [cite_start]3. 객체 분산도 점수 (최대 25점) [cite: 255, 257, 258]
    all_x = [d['norm_x_center'] for d in detections]
    if len(all_x) > 1:
        spread_score = np.std(all_x) * 100 
        score += min(25, spread_score)
        
    return max(0, min(100, score))

def calculate_warmth(detections, img_w, img_h):
    # [cite_start]온화성 점수 계산 [cite: 263]
    score = 0
    sun_boxes = [d for d in detections if d['label'] == '태양']
    chimney_boxes = [d for d in detections if d['label'] == '굴뚝']
    house_boxes = [d for d in detections if d['label'] == '집전체']
    flower_count = len([d for d in detections if d['label'] == '꽃'])
    
    # [cite_start]1. 태양 면적 점수 (최대 70점) [cite: 270, 271]
    if sun_boxes:
        sun = sun_boxes[0]
        sun_area = (sun['pixel_x2'] - sun['pixel_x1']) * (sun['pixel_y2'] - sun['pixel_y1'])
        sun_ratio = sun_area / (img_w * img_h)
        score += min(70, sun_ratio * 100000) # (임의의 스케일링)
        
    # [cite_start]2. 굴뚝 면적 점수 (최대 30점) [cite: 268, 269]
    if chimney_boxes and house_boxes:
        chimney = chimney_boxes[0]
        main_house = max(house_boxes, key=lambda d: (d['pixel_x2'] - d['pixel_x1']) * (d['pixel_y2'] - d['pixel_y1']))
        chimney_area = (chimney['pixel_x2'] - chimney['pixel_x1']) * (chimney['pixel_y2'] - chimney['pixel_y1'])
        house_area = (main_house['pixel_x2'] - main_house['pixel_x1']) * (main_house['pixel_y2'] - main_house['pixel_y1'])
        
        if house_area > 0:
            chimney_ratio = chimney_area / house_area
            score += min(30, chimney_ratio * 200) # (임의의 스케일링)
            
    # [cite_start]3. 꽃 개수 점수 (최대 30점) [cite: 272, 274]
    score += min(30, flower_count * 5)
    
    return max(0, min(100, score))

def calculate_stability(detections, img_w, img_h):
    # [cite_start]현실 기반 안정성 점수 계산 [cite: 278]
    score = 0
    roof_boxes = [d for d in detections if d['label'] == '지붕']
    house_boxes = [d for d in detections if d['label'] == '집전체']
    smoke_boxes = [d for d in detections if d['label'] == '연기']
    
    if not house_boxes: return 50
    main_house = max(house_boxes, key=lambda d: (d['pixel_x2'] - d['pixel_x1']) * (d['pixel_y2'] - d['pixel_y1']))
    house_area = (main_house['pixel_x2'] - main_house['pixel_x1']) * (main_house['pixel_y2'] - main_house['pixel_y1'])
    
    # [cite_start]1. 지붕 면적 점수 (최대 50점) [cite: 281, 282, 283]
    if roof_boxes:
        roof = roof_boxes[0]
        roof_area = (roof['pixel_x2'] - roof['pixel_x1']) * (roof['pixel_y2'] - roof['pixel_y1'])
        if house_area > 0:
            roof_ratio = roof_area / house_area
            # 0.4~0.6 범위에 가까울수록 점수 상승
            score += max(0, 50 - abs(roof_ratio - 0.5) * 100)
    
    # [cite_start]2. 집 중심 위치 점수 (최대 30점) [cite: 284, 285, 286]
    house_cx = main_house['pixel_x1'] + (main_house['pixel_x2'] - main_house['pixel_x1']) / 2
    house_cy = main_house['pixel_y1'] + (main_house['pixel_y2'] - main_house['pixel_y1']) / 2
    
    center_dist_norm = np.sqrt((house_cx - img_w/2)**2 + (house_cy - img_h/2)**2) / (np.sqrt(img_w**2 + img_h**2) / 2)
    score += 30 * (1 - center_dist_norm)
    
    # [cite_start]3. 연기 면적 점수 (최대 20점) [cite: 290, 291, 292]
    if smoke_boxes:
        smoke = smoke_boxes[0]
        smoke_area = (smoke['pixel_x2'] - smoke['pixel_x1']) * (smoke['pixel_y2'] - smoke['pixel_y1'])
        smoke_ratio = smoke_area / (img_w * img_h)
        if smoke_ratio <= 0.05:
            score += 20 * (1 - smoke_ratio / 0.05)
            
    # [cite_start]4. 좌우 치우침 감점 (최대 -15점) [cite: 293, 294, 295]
    center_dist_x = abs(house_cx - img_w/2)
    img_w_20_percent = img_w * 0.2
    if center_dist_x > img_w_20_percent:
        # 치우친 정도에 비례하여 최대 15점 감점
        penalty_ratio = (center_dist_x - img_w_20_percent) / (img_w/2 - img_w_20_percent)
        score -= min(15, 15 * penalty_ratio)
    
    return max(0, min(100, score))

def get_scores(detections, img_w=1280, img_h=1280):
    """주요 호출 함수"""
    scores = {
        'openness': calculate_openness(detections, img_w, img_h),
        'vitality': calculate_vitality(detections, img_w, img_h),
        'warmth': calculate_warmth(detections, img_w, img_h),
        'stability': calculate_stability(detections, img_w, img_h)
    }
    return scores