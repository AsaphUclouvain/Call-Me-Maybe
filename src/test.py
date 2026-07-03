import json
from pydantic import BaseModel

data = """
{
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
        "a": {
        "type": "number"
        },
        "b": {
        "type": "number"
        }
    },
    "returns": {
        "type": "number"
    }
}
"""

class VarMetaData(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, VarMetaData]
    returns: VarMetaData

d = json.loads(data)
print(FunctionDef(**d))