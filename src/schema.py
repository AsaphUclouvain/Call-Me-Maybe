"""Pydantic models describing function metadata and user inputs."""

from pydantic import BaseModel
from typing import Literal


class VarMetaData(BaseModel):
    """Metadata for a function parameter or return value."""
    type: Literal["number", "string", "boolean"]


class FunctionDef(BaseModel):
    """Definition of an available function the model may call."""
    name: str
    description: str
    parameters: dict[str, VarMetaData]
    returns: VarMetaData


class UserInput(BaseModel):
    """A single user prompt to be translated into a function call."""
    prompt: str
