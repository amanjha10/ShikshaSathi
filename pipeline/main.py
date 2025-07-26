"""
Main entry point for the Voice-to-Voice application pipeline
Structured FastAPI server with modular handlers
"""

import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import base64
import json
import os
from tts import TTSHandler
from stt import STTHandler
from rag import RAGHandler

# Translation layer removed - now using direct multilingual embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tts_handler = TTSHandler()
stt_handler = STTHandler()
rag_handler = RAGHandler()

app = FastAPI(title="Voice-to-Voice Chat", version="1.0.0")
app.mount("/static", StaticFiles(directory=".."), name="static")

@app.get("/")
async def get_client():
    """Serve the web client"""
    return HTMLResponse(open(os.path.join("..", "voice_chat_client.html")).read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"New client connected: {websocket.client}")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "audio":
                try:
                    audio_data = base64.b64decode(message["audio"])
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Processing your voice message..."
                    }))
                    text = stt_handler.transcribe(audio_data)
                    logger.info(f"STT output: {text}")
                    # Always send buffer message for ALL queries to maintain consistent perceived latency
                    buffer_message = "तपाईंको प्रश्नको लागि धन्यवाद। जवाफ तयार पार्दै छु..."
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": buffer_message,
                        "is_buffer": True
                    }))
                    # Also send buffer message as audio for immediate playback
                    buffer_audio = tts_handler.synthesize(buffer_message)
                    buffer_audio_b64 = base64.b64encode(buffer_audio).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": buffer_audio_b64,
                        "is_buffer": True
                    }))
                    # Direct Nepali query to RAG (no translation needed)
                    nepali_response = rag_handler.get_nepali_answer(text)
                    logger.info(f"RAG Nepali response: {nepali_response}")
                    audio_response = tts_handler.synthesize(nepali_response)
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "is_buffer": False
                    }))
                    response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": response_audio_b64
                    }))
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}"
                    }))
            elif message.get("type") == "text":
                try:
                    input_text = message.get("text", "")
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Processing your question..."
                    }))
                    logger.info(f"Query to RAG (from text): {input_text}")
                    # Always send buffer message for ALL queries to maintain consistent perceived latency
                    buffer_message = "तपाईंको प्रश्नको लागि धन्यवाद। जवाफ तयार पार्दै छु..."
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": buffer_message,
                        "is_buffer": True
                    }))
                    # Also send buffer message as audio for immediate playback
                    buffer_audio = tts_handler.synthesize(buffer_message)
                    buffer_audio_b64 = base64.b64encode(buffer_audio).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": buffer_audio_b64,
                        "is_buffer": True
                    }))
                    # Direct Nepali query to RAG (no translation needed)
                    nepali_response = rag_handler.get_nepali_answer(input_text)
                    logger.info(f"RAG Nepali response: {nepali_response}")
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "is_buffer": False
                    }))
                    audio_response = tts_handler.synthesize(nepali_response)
                    if audio_response:
                        response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                        await websocket.send_text(json.dumps({
                            "type": "audio_response",
                            "audio": response_audio_b64
                        }))
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Text processing error: {str(e)}"
                    }))
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "message": "Server is alive"
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Unknown message type"
                }))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    logger.info("🎙️  Starting Voice-to-Voice FastAPI Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
