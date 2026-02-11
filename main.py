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
app = FastAPI(title="DeepVoice Detect API", version="1.0")

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

# --- Audio Processing Logic ---
def analyze_audio_signal(audio_bytes):
    try:
        # Decode Base64 to Audio
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=None) 
        
        # Extract Features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        variance = np.var(y)
        
        # Heuristic Logic
        explanation_parts = []
        score = 0.5 
        
        if variance < 0.0005:
            score += 0.3
            explanation_parts.append(f"Signal lacks dynamic range (Variance: {variance:.4f}).")
        else:
            explanation_parts.append(f"Signal shows natural dynamic range (Variance: {variance:.4f}).")

        if flatness < 0.01:
            score += 0.15
            explanation_parts.append(f"Spectral flatness ({flatness:.4f}) suggests synthetic consistency.")
        
        classification = "AI GENERATED" if (score > 0.6 or variance < 0.0001) else "HUMAN"
        final_confidence = min(max(score, 0.1), 0.99)
        
        full_explanation = " ".join(explanation_parts)
        if not full_explanation:
            full_explanation = f"Audio features are within standard parameters. ZCR: {zcr:.3f}."

        return classification, final_confidence, full_explanation

    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")

# --- API Endpoint (The core requirement) ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_voice(request: AudioRequest):
    try:
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")

        classification, confidence, explanation = analyze_audio_signal(audio_bytes)

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
        return {
            "status": "error",
            "language": request.language,
            "classification": "UNKNOWN",
            "confidenceScore": 0.0,
            "explanation": f"Internal processing error: {str(e)}"
        }

# --- NEW: Built-in UI for Easy Testing ---
@app.get("/", response_class=HTMLResponse)
async def get_testing_ui():
    """
    This provides a clean web interface so you don't have to manually 
    copy-paste massive Base64 strings to test the API.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Tester</title>
        <style>
            body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            input, select, button { margin-top: 10px; width: 100%; padding: 10px; box-sizing: border-box; }
            button { background-color: #007bff; color: white; border: none; cursor: pointer; font-weight: bold; }
            button:hover { background-color: #0056b3; }
            pre { background: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h2>Audio API Tester</h2>
        <p>Select an audio file. This page will automatically convert it to Base64 and send the correct JSON to the <code>/analyze</code> endpoint.</p>
        
        <label><b>Language:</b></label>
        <select id="language">
            <option value="Hindi">Hindi</option>
            <option value="English">English</option>
            <option value="Tamil">Tamil</option>
            <option value="Malayalam">Malayalam</option>
            <option value="Telugu">Telugu</option>
        </select>
        
        <label style="display:block; margin-top:15px;"><b>Select Audio File:</b></label>
        <input type="file" id="audioFile" accept="audio/*">
        
        <button onclick="sendData()" style="margin-top:20px;">Test API</button>
        
        <h3 style="margin-top:30px;">JSON Response:</h3>
        <pre id="result">Waiting for input...</pre>

        <script>
            async function sendData() {
                const fileInput = document.getElementById('audioFile');
                const langInput = document.getElementById('language').value;
                const resultBox = document.getElementById('result');
                
                if (!fileInput.files.length) {
                    alert("Please select an audio file first.");
                    return;
                }

                resultBox.innerText = "Processing audio... Please wait.";
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = async function(event) {
                    // Strip the "data:audio/mp3;base64," prefix to get raw Base64
                    const base64String = event.target.result.split(',')[1];
                    
                    const payload = {
                        audio_base64: base64String,
                        language: langInput
                    };

                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        
                        const data = await response.json();
                        resultBox.innerText = JSON.stringify(data, null, 2);
                    } catch (error) {
                        resultBox.innerText = "Error connecting to API: " + error.message;
                    }
                };
                reader.readAsDataURL(file);
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)