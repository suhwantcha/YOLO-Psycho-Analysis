def get_analysis_label(scores):
    """
    4가지 지표 점수를 바탕으로 가장 두드러지는 상위 2가지 특성을 분류합니다.
    (우선순위: O > V > W > S) [cite_start][cite: 306]
    """
    # [cite_start]각 점수와 기준점(50)의 차이 (절댓값) 계산 [cite: 304, 305]
    deltas = {k: abs(v - 50) for k, v in scores.items()}
    
    # [cite_start]우선순위: openness > vitality > warmth > stability [cite: 307]
    priority = ["openness", "vitality", "warmth", "stability"]

    # [cite_start]가장 두드러지는 두 특성 추출: 1) 50점과의 차이가 큰 순, 2) 우선순위가 높은 순 [cite: 308, 309, 310]
    top2 = sorted(priority, key=lambda k: (-deltas[k], priority.index(k)))[:2]
    
    # [cite_start]점수에 따라 대문자(>=50) 또는 소문자(<50) 적용하여 최종 라벨 생성 [cite: 311]
    label = "".join(k[0].upper() if scores[k] >= 50 else k[0].lower() for k in top2)
    return label
    
def get_script(label, script_data):
    return script_data.get(label, "분석 결과에 대한 스크립트가 준비되지 않았습니다.")