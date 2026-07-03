import json
from pydantic import BaseModel, ValidationError
from pathlib import Path
from typing import Literal


class VarMetaData(BaseModel):
    type: Literal["number", "string"]


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, VarMetaData]
    returns: VarMetaData

def load_function_from_json(file_path: str) -> list[dict[str, str | dict]]:
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8")
        raw_data = json.loads(text)
        if not isinstance(raw_data, list):
            raise ValueError(f"le fichier {file_path} doit contenir une liste JSON.")
        return [FunctionDef(**f) for f in raw_data]
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as e:
        raise ValueError(
                f"Impossible de charger les donnees depuis {file_path}: {e}"
            ) from e
    
print(load_function_from_json("data/input/functions_definition.json"))