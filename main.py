import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Echolyze Bio-Forensic API", version="3.0")

class AudioRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = "Unknown"

class AnalysisResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

def calculate_biometrics(y, sr):
    """
    Extracts Jitter (pitch instability) and Shimmer (amplitude instability).
    Humans have higher Jitter/Shimmer than AI.
    """
    try:
        # 1. Extract Pitch (F0) using probabilistic YIN
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Filter only voiced parts (remove silence)
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) < 10:
            return 0.0, 0.0, 0.0 # Audio too short or unvoiced

        # 2. Calculate Jitter (Frequency Instability)
        # Average absolute difference between consecutive pitch periods
        jitter = np.mean(np.abs(np.diff(f0))) / np.mean(f0)

        # 3. Calculate Shimmer (Amplitude Instability)
        # We need the RMS amplitude of the voiced frames
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        # Resize RMS to match F0 length if needed
        min_len = min(len(rms), len(f0))
        rms = rms[:min_len]
        
        # Calculate Shimmer
        shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
        
        # 4. High Frequency Cepstral Coeffs (Synthetic Texture)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_var = np.var(mfcc[1:]) # Skip the first coefficient (energy)
        
        return jitter, shimmer, mfcc_var

    except Exception as e:
        print(f"Bio-Analysis Error: {e}")
        return 0.0, 0.0, 0.0

def analyze_audio_forensics(audio_bytes):
    try:
        # Load Audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")

        # --- Extract Biometric Features ---
        jitter, shimmer, mfcc_texture = calculate_biometrics(y, sr)
        
        # --- Extract Signal Features ---
        # AI often has lower spectral rolloff (cuts frequencies early)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        
        # --- SCORING ENGINE (The Brain) ---
        # We calculate a 'Liveness Score'. Higher = More likely Human.
        
        liveness_score = 0.0
        details = []

        # 1. Jitter Analysis (Human vocal cords vibrate irregularly)
        # Typical Human Jitter > 0.01. AI is often < 0.005
        if jitter > 0.008: 
            liveness_score += 0.35
            details.append(f"High pitch fluctuation (Jitter: {jitter:.4f}) indicates organic vocal cords.")
        else:
            liveness_score -= 0.30
            details.append(f"Unnaturally stable pitch (Jitter: {jitter:.4f}) suggests algorithmic synthesis.")

        # 2. Shimmer Analysis (Breath/Volume control)
        # Humans vary volume with breath. AI is mathematically consistent.
        if shimmer > 0.15:
            liveness_score += 0.25
            details.append(f"Natural amplitude variation detected (Shimmer: {shimmer:.3f}).")
        else:
            liveness_score -= 0.15
            details.append(f"Amplitude is too consistent (Shimmer: {shimmer:.3f}).")

        # 3. Frequency Analysis (The 11kHz/16kHz Cutoff)
        # Real human speech (on standard mic) usually has rolloff > 3000Hz
        if rolloff > 3500:
            liveness_score += 0.20
            details.append("Full frequency spectrum present.")
        elif rolloff < 2000:
            liveness_score -= 0.20
            details.append("High frequencies are artificially clipped.")

        # 4. MFCC Texture (Robotic vs Rich)
        if mfcc_texture > 1500: # High variance = Rich human voice
             liveness_score += 0.20
        else:
             liveness_score -= 0.20

        # --- DYNAMIC CONFIDENCE CALCULATION ---
        # Normalize score to -1.0 to 1.0 range roughly
        
        if liveness_score > 0:
            classification = "HUMAN"
            # Calculate how sure we are (0.50 to 0.99)
            # We map the score (0.0 to 1.0) to confidence
            raw_confidence = 0.5 + (min(liveness_score, 1.0) / 2)
            confidence = min(max(raw_confidence, 0.55), 0.99)
            main_reason = "Bio-metric analysis detects natural vocal cord irregularities."
        else:
            classification = "AI GENERATED"
            # Map negative score to confidence
            raw_confidence = 0.5 + (min(abs(liveness_score), 1.0) / 2)
            confidence = min(max(raw_confidence, 0.55), 0.99)
            main_reason = "Signal lacks the micro-tremors inherent to biological speech production."

        # Final Explanation
        final_explanation = f"{main_reason} Evidence: {' '.join(details)}"
        
        return classification, round(confidence, 3), final_explanation

    except Exception as e:
        return "UNKNOWN", 0.0, f"Error: {str(e)}"

# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <body style="font-family: sans-serif; text-align: center; padding: 50px;">
        <h1>Echolyze Bio-Forensic API</h1>
        <p style="color: green;">● System Operational</p>
        <p>Send POST requests to <b>/analyze</b></p>
    </body>
    </html>
    """

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AudioRequest):
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="Missing audio")
        
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        cls, conf, exp = analyze_audio_forensics(audio_bytes)
        
        return {
            "status": "success",
            "language": request.language,
            "classification": cls,
            "confidenceScore": conf,
            "explanation": exp
        }
    except Exception as e:
        return {
            "status": "error", 
            "language": request.language, 
            "classification": "ERROR", 
            "confidenceScore": 0.0, 
            "explanation": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)