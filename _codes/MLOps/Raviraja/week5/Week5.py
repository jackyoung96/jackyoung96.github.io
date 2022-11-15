from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from NLPmodel import ColaPredictor

app = FastAPI(title="MLOps Basic App")
predictor = ColaPredictor("models/model.onnx")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <html>
        <h2>This is a sample NLP Project</h2>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

@app.get("/predict")
async def get_prediction(text: str):
    result = predictor.predict(text)
    return result

if __name__=='__main__':
    uvicorn.run("Week5:app", host="147.46.92.196", port=8000)