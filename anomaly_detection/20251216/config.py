# src/anomaly_detection/config.py
import yaml
import os


def load_config(filepath):
    """
    YAML 설정 파일을 로드하고, ${key} 형식의 변수를 치환한 후 모든 경로를 os.path 기반 절대 경로로 변환하여 반환
    """
    # 1. YAML 파일 로드
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. ${key} 형식의 변수 재귀적 치환 (최대 10회 반복 - 순환 참조 방지)
    for _ in range(10):
        updated = False
        for key, value in config.items():
            if isinstance(value, str):
                original = value

                for k, v in config.items():
                    if isinstance(v, (str, int, float)):
                        value = value.replace(f"${{{k}}}", str(v))
                config[key] = value
                if original != value:
                    updated = True
        if not updated:
            break

    # 3. dir/path 포함 키는 os.path.abspath로 절대 경로 변환
    for key, value in config.items():
        if isinstance(value, str) and ('dir' in key.lower() or 'path' in key.lower()):
            config[key] = os.path.abspath(value)

    return config
