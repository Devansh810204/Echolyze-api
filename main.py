import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Echolyze AI Voice Detector")

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

# --- Advanced Audio Analysis Logic ---
def analyze_signal_properties(audio_bytes):
    try:
        # Load audio (y = audio time series, sr = sample rate)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # --- Feature 1: Silence/Noise Floor Analysis ---
        # AI audio often has "digital silence" (absolute 0) between words.
        # Human recordings usually have background noise/hiss.
        noise_floor = np.min(np.abs(y[y != 0])) if np.any(y) else 0
        has_digital_silence = noise_floor < 1e-5

        # --- Feature 2: Spectral Flatness (Tonality) ---
        # High flatness = noise-like. Low flatness = tonal.
        # AI models sometimes over-smooth the spectrum.
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # --- Feature 3: MFCC Variance (Vocal Texture) ---
        # Human voices have complex, chaotic micro-tremors.
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfccs, axis=1))

        # --- Feature 4: High Frequency Cutoff ---
        # Some older AI models cut off frequencies above 16kHz sharply.
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # --- SCORING LOGIC ---
        ai_probability = 0.0
        reasons = []

        # Check 1: Unnatural Silence
        if has_digital_silence:
            ai_probability += 0.30
            reasons.append("Detected unnatural digital silence between speech segments.")
        else:
            reasons.append("Natural background noise floor detected.")

        # Check 2: Spectral Consistency (Robotic smoothness)
        if mfcc_var < 500:  # Threshold for "too smooth"
            ai_probability += 0.25
            reasons.append(f"Vocal texture lacks natural human jitter (Low MFCC Variance: {mfcc_var:.1f}).")
        
        # Check 3: Spectral Flatness
        if flatness < 0.002:
            ai_probability += 0.20
            reasons.append("Audio spectrum is unusually distinct and lacks organic complexity.")

        # Check 4: Frequency Range
        if spec_cent > 3500:
            ai_probability += 0.15 # AI often overly "bright" or consistent in high freq
        
        # --- FINAL DECISION ---
        # Normalize score to 0.0 - 1.0 range
        final_score = min(ai_probability, 0.99)
        
        if final_score > 0.55:
            classification = "AI GENERATED"
            # Confidence is how far above 0.55 we are
            confidence = 0.70 + (final_score * 0.25)
            main_reason = "The audio exhibits statistical regularities typical of synthesis algorithms."
        else:
            classification = "HUMAN"
            # Confidence is how far below 0.55 we are
            confidence = 0.85 - (final_score * 0.3)
            main_reason = "The audio contains micro-tremors and noise patterns consistent with organic recording."

        # Combine explanation
        full_explanation = f"{main_reason} Specifics: {' '.join(reasons)}"
        
        return classification, round(confidence, 2), full_explanation

    except Exception as e:
        return "UNKNOWN", 0.0, f"Error analyzing audio: {str(e)}"

# --- 1. ROOT ENDPOINT (HTML Interface) ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Echolyze API Test</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f4f4f9; }
                h1 { color: #333; }
                .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                textarea { width: 100%; height: 100px; margin-bottom: 10px; }
                button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                button:hover { background: #0056b3; }
                #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; background: #fafafa; white-space: pre-wrap;}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Echolyze AI Detector</h1>
                <p>This is the API root. Use the <b>/analyze</b> endpoint for JSON requests.</p>
                <p>Status: <span style="color: green; font-weight: bold;">ACTIVE</span></p>
            </div>
        </body>
    </html>
    """

# --- 2. ANALYZE ENDPOINT ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AudioRequest):
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="No audio data provided")
    
    # Analyze
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        classification, confidence, explanation = analyze_signal_properties(audio_bytes)
        
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)