import base64
import io
import json
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI
app = FastAPI(title="Echolyze API", version="1.0")

# --- Data Models ---
class AudioRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = "Unknown" 

class AnalysisResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

# --- 1. ROOT ENDPOINT (Fixes the 404 Error) ---
@app.get("/")
def home():
    return {
        "status": "active",
        "message": "Echolyze API is running successfully.",
        "instructions": "Send a POST request to /analyze with {'audio_base64': '...', 'language': '...'}"
    }

# --- Helper: Feature Extraction & Heuristic Logic ---
def analyze_audio_signal(audio_bytes):
    try:
        # Decode Base64 to Audio
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=None) 
        
        # Extract Features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        variance = np.var(y)
        
        # Heuristic Logic (Simulation for the API)
        explanation_parts = []
        score = 0.5 
        is_ai = False

        # Check Variance (AI often lacks dynamic range)
        if variance < 0.0005:
            score += 0.3
            explanation_parts.append(f"Signal lacks dynamic range (Variance: {variance:.4f}).")
            is_ai = True
        else:
            explanation_parts.append(f"Signal shows natural dynamic range.")

        # Check Spectral Flatness (AI often has consistent flatness)
        if flatness < 0.01:
            score += 0.15
            explanation_parts.append(f"Spectral flatness ({flatness:.4f}) suggests synthetic consistency.")
            is_ai = True if score > 0.6 else is_ai
        
        # Final Decision
        classification = "AI GENERATED" if (score > 0.65 or variance < 0.0001) else "HUMAN"
        final_confidence = min(max(score, 0.1), 0.99)
        
        full_explanation = " ".join(explanation_parts)
        if not full_explanation:
            full_explanation = f"Audio features are within standard parameters. ZCR: {zcr:.3f}."

        return classification, final_confidence, full_explanation

    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")

# --- 2. ANALYZE ENDPOINT (The Main Tool) ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_voice(request: AudioRequest):
    try:
        # Decode Base64
        if not request.audio_base64:
             raise HTTPException(status_code=400, detail="Audio data is missing")
             
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")

        # Analyze
        classification, confidence, explanation = analyze_audio_signal(audio_bytes)

        return {
            "status": "success",
            "language": request.language if request.language else "Detected",
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        return {
            "status": "error",
            "language": request.language,
            "classification": "UNKNOWN",
            "confidenceScore": 0.0,
            "explanation": f"Internal processing error: {str(e)}"
        }

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)