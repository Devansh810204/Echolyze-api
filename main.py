import base64
import io
import json
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI
app = FastAPI(title="DeepVoice Detect API", version="1.0")

# --- Data Models ---
class AudioRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = "Unknown"  # Input language as per prompt

class AnalysisResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

# --- Helper: Feature Extraction & Mock Classification ---
def analyze_audio_signal(audio_bytes):
    """
    Decodes audio and extracts features (ZCR, MFCCs, Flatness).
    In a production system, these features would be fed into a trained 
    TensorFlow/PyTorch model. Here, we use a heuristic approach.
    """
    try:
        # 1. Decode Base64 to Audio
        audio_file = io.BytesIO(audio_bytes)
        # Load audio (y=waveform, sr=sample rate)
        y, sr = librosa.load(audio_file, sr=None) 
        
        # 2. Extract Features
        # Zero Crossing Rate (ZCR): High ZCR often indicates noise or fricatives
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Spectral Flatness: AI audio often has different flatness characteristics
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Variance/Dynamic Range: AI models sometimes struggle with natural amplitude variance
        variance = np.var(y)
        
        # RMS Energy: Loudness
        rms = np.mean(librosa.feature.rms(y=y))

        # 3. Logic / Classification (Heuristic Placeholder)
        # NOTE: Replace this logic with: prediction = model.predict(features)
        
        explanation_parts = []
        score = 0.5  # Base score
        is_ai = False

        # Heuristic 1: Unnaturally low variance (flat sounding)
        if variance < 0.0005:
            score += 0.3
            explanation_parts.append(f"Signal lacks dynamic range (Variance: {variance:.4f}).")
            is_ai = True
        else:
            explanation_parts.append(f"Signal shows natural dynamic range (Variance: {variance:.4f}).")

        # Heuristic 2: Spectral Flatness (Synthetic audio can sometimes be too 'pure')
        if flatness < 0.01:
            score += 0.15
            explanation_parts.append(f"Spectral flatness ({flatness:.4f}) suggests synthetic consistency.")
            is_ai = True if score > 0.6 else is_ai
        
        # Heuristic 3: Check for absolute silence (digital silence is rare in human recordings)
        if np.min(np.abs(y)) == 0 and np.std(y) > 0.01:
             explanation_parts.append("Digital silence detected in gaps.")

        # Final Classification Decision
        classification = "AI GENERATED" if (score > 0.6 or variance < 0.0001) else "HUMAN"
        
        # Normalize confidence score (0.0 to 1.0)
        final_confidence = min(max(score, 0.1), 0.99)
        
        # Construct Explanation
        full_explanation = " ".join(explanation_parts)
        if not full_explanation:
            full_explanation = f"Audio features are within standard parameters. ZCR: {zcr:.3f}."

        return classification, final_confidence, full_explanation

    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_voice(request: AudioRequest):
    try:
        # Decode Base64 string
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")

        # Perform Analysis
        classification, confidence, explanation = analyze_audio_signal(audio_bytes)

        # Build Response
        return {
            "status": "success",
            "language": request.language if request.language else "Detected from Audio",
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Log error in real app
        return {
            "status": "error",
            "language": request.language,
            "classification": "UNKNOWN",
            "confidenceScore": 0.0,
            "explanation": f"Internal processing error: {str(e)}"
        }

# For testing locally without running uvicorn command
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)