from pydantic import BaseModel
from typing import Literal

class VarMetaData(BaseModel):
    type: Literal["number", "string", "boolean"]

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, VarMetaData]
    returns: VarMetaData

class UserInput(BaseModel):
    prompt: str
