import os
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from inference import CodiInference


class ResultCategory(BaseModel):
    category: str


codi_inference = CodiInference('./configs.yaml')

app = FastAPI()


@app.post('/predict/category', response_model=ResultCategory)
def predict_category(image_path: str):
    try:
        assert os.path.isfile(image_path), 'Not a file.'
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"""{image_path} is incorrect path, {e}"""
        )
    cat = codi_inference.inference(image_path)
    result = ResultCategory(category=cat)
    return result
