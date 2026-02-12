import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from scipy.stats import entropy

app = FastAPI(title="Echolyze AI Voice Detector", version="2.0")

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

# --- Advanced Forensic Analysis ---
def forensic_analysis(audio_bytes):
    try:
        # 1. Load Audio
        # We use a specific sample rate (sr=None) to capture the original quality
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # 2. Safety Check for empty/short audio
        if len(y) < sr * 0.5: # Less than 0.5 seconds
            return "UNKNOWN", 0.0, "Audio is too short for reliable forensic analysis."

        # --- FEATURE 1: High Frequency Bandwidth (The "Cutoff" Test) ---
        # Real human voices have energy all the way up to 20kHz (if recorded on good mics).
        # Many AI models (Tacotron/WaveNet) cut off sharply at 11kHz or 8kHz.
        spec = np.abs(librosa.stft(y))
        # Calculate energy in high freq bands (above 11kHz)
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_idx = np.where(freqs > 11000)[0]
        if len(high_freq_idx) > 0:
            high_freq_energy = np.mean(spec[high_freq_idx, :])
            total_energy = np.mean(spec)
            hf_ratio = high_freq_energy / (total_energy + 1e-6)
        else:
            hf_ratio = 0.0

        # --- FEATURE 2: Digital Silence (The "Zero" Test) ---
        # Real microphones have "noise floor" (air hiss). AI generates absolute zeros.
        # We check the percentage of samples that are EXACTLY 0.0
        zero_samples = np.sum(y == 0)
        total_samples = len(y)
        silence_ratio = zero_samples / total_samples

        # --- FEATURE 3: Spectral Entropy (Complexity) ---
        # Human voice is chaotic and complex (High Entropy).
        # AI voice is mathematically generated and more ordered (Low Entropy).
        # We normalize the power spectrum and calculate Shannon entropy
        power_spec = np.mean(spec, axis=1)
        power_spec_norm = power_spec / (np.sum(power_spec) + 1e-9)
        spec_entropy = entropy(power_spec_norm)

        # --- SCORING ENGINE ---
        ai_score = 0.0
        reasons = []

        # 1. Analyze Silence (Strong Indicator)
        if silence_ratio > 0.05: # More than 5% absolute silence
            ai_score += 0.40
            reasons.append(f"Contains unnatural digital silence ({silence_ratio*100:.1f}% samples are perfect zeros).")
        
        # 2. Analyze High Frequencies
        # Humans usually have hf_ratio > 0.05 on standard mics. AI is often < 0.01
        if hf_ratio < 0.002: 
            ai_score += 0.35
            reasons.append("Severe lack of high-frequency information (Audio cuts off unnaturally).")
        elif hf_ratio < 0.01:
            ai_score += 0.15
            reasons.append("Weak high-frequency presence typical of upsampled AI audio.")

        # 3. Analyze Entropy (The "Texture" Test)
        # Typical Human Entropy > 5.0 (depending on bin count, but relative is key)
        # If entropy is suspiciously low, it's likely synthetic.
        if spec_entropy < 4.0: 
            ai_score += 0.25
            reasons.append(f"Spectral complexity is low ({spec_entropy:.2f}), suggesting algorithmic generation.")

        # --- FINAL CLASSIFICATION ---
        
        # Base decision threshold
        if ai_score > 0.45:
            classification = "AI GENERATED"
            # Confidence calculation: Map score 0.45-1.0 to 75%-99%
            confidence = min(0.75 + (ai_score - 0.45), 0.99)
            main_explanation = "The audio signal lacks the acoustic complexity of organic speech."
        else:
            classification = "HUMAN"
            # Confidence calculation: Map score 0.45-0.0 to 70%-98%
            confidence = min(0.70 + (0.45 - ai_score), 0.98)
            main_explanation = "The audio contains natural noise floors and frequency richness consistent with human recording."

        # Fallback if no specific reasons found but score was low
        if not reasons and classification == "AI GENERATED":
            reasons.append("Signal statistical properties align with synthetic training data.")

        final_explanation = f"{main_explanation} Key factors: {' '.join(reasons)}"

        return classification, round(confidence, 2), final_explanation

    except Exception as e:
        # Fallback for corrupted audio
        return "UNKNOWN", 0.0, f"Error during forensic analysis: {str(e)}"

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return 

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AudioRequest):
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="Audio data is missing")
    
    try:
        # Decode Base64
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")

        # Perform Forensic Analysis
        classification, confidence, explanation = forensic_analysis(audio_bytes)
        
        return {
            "status": "success",
            "language": request.language if request.language else "Detected",
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "language": request.language,
            "classification": "UNKNOWN",
            "confidenceScore": 0.0,
            "explanation": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)