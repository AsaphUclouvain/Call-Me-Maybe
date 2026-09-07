from pathlib import Path
import config
import json

def read_json(file_name: str) -> list | dict:
    path = Path(file_name)
    try:
        text = path.read_text()
        return json.loads(text)
    except (OSError, json.JSONDecodeError)  as e:
        raise ValueError(f"impossible de charger les donnees depuis {file_name}") from e

def write_json(json_data: str) -> str:
    path = Path(config.OUTPUT_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_data)
    except (OSError, json.JSONDecodeError)  as e:
        raise ValueError(f"impossible d'ecrie {config.OUTPUT_FILE}") from e
