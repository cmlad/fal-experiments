from pydantic import BaseModel
from fal.toolkit import Image

class Input(BaseModel):
    prompt: str

class Output(BaseModel):
    images: list[Image]
