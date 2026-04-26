# SpiceCheck API

Food adulteration detection backend — EfficientNet-B3 + GradCAM  
B.E. Final Year Project | CMR Institute of Technology

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Model status |
| POST | `/predict` | Predict + GradCAM heatmap |
| POST | `/predict/fast` | Predict only (no heatmap) |

## Local development

```bash
# 1. Put your model files in this folder
#    - efficientnet_b3_best.pth
#    - class_names.json

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
uvicorn main:app --reload --port 8000

# 4. Open API docs
# http://localhost:8000/docs
```

## Test with curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_chilli_photo.jpg"
```

## Example response

```json
{
  "prediction": "chilli_brick_high",
  "label": "Heavily Adulterated",
  "confidence": 99.4,
  "adulterant": "Brick powder",
  "pct_range": "30-50%+",
  "safe": false,
  "fssai_advisory": "Heavy brick powder adulteration. Serious FSSAI violation...",
  "heatmap_base64": "iVBORw0KGgoAAAANS...",
  "inference_time_ms": 1823.4
}
```

## Deploy to Render.com

1. Push this folder to GitHub
2. Go to render.com → New Web Service → connect your repo
3. Add environment variables from render.yaml
4. Deploy
