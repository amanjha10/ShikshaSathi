import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
import json
import os
import numpy as np
import soundfile as sf
import io
from dotenv import load_dotenv
from agent import Agent
from stt_handler import STTHandler
from tts_handler import TTSHandler
from rag_handler import RAGHandler
from vad_handler import VADHandler, AudioProcessor
# Translation layer removed - using direct multilingual embeddings
# Using RAG handler's built-in Nepali summarization

# Load environment variables from .env file (look in parent directories)
load_dotenv(dotenv_path="../../.env")
load_dotenv()  # Also try current directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Call Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Initialize handlers
agent = Agent()
stt_handler = STTHandler()
tts_handler = TTSHandler()
rag_handler = RAGHandler()

# Store VAD handlers per client
client_vad_handlers = {}

@app.get("/")
async def root():
    with open(os.path.join(os.path.dirname(__file__), "../frontend/index.html")) as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"New client connected: {websocket.client} (ID: {client_id})")

    # Initialize VAD handler for this client with higher threshold for loud speech
    vad_handler = VADHandler(
        energy_threshold=0.08,  # Higher threshold - need to speak louder
        silence_duration=1.5,   # 1.5 seconds of silence to end speech
        min_speech_duration=0.5  # Minimum 0.5 seconds for valid speech
    )
    audio_processor = AudioProcessor(chunk_duration=0.1, sample_rate=16000)
    client_vad_handlers[client_id] = {
        'vad': vad_handler,
        'processor': audio_processor,
        'current_time': 0.0
    }

    # Send model status on connect
    await websocket.send_text(json.dumps({
        "type": "status",
        "models": {
            "stt": True,
            "tts": True,
            "rag": rag_handler.retriever is not None,
            "gemini": True
        },
        "vad_enabled": True
    }))

    # Send welcome message
    welcome_message = "नमस्ते! काठमाडौं विश्वविद्यालयको भर्ना सहायकको रूपमा, तपाईंलाई भर्ना, कोर्स, शुल्क वा अध्ययनबारे केही सोध्नु छ?"
    await websocket.send_text(json.dumps({
        "type": "text_response",
        "text": welcome_message,
        "is_welcome": True
    }))
    # Also send welcome message as audio
    welcome_audio = tts_handler.synthesize(welcome_message)
    welcome_audio_b64 = base64.b64encode(welcome_audio).decode('utf-8')
    await websocket.send_text(json.dumps({
        "type": "audio_response",
        "audio": welcome_audio_b64,
        "is_welcome": True
    }))
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            step_log = []
            agent_decision = None
            rag_context = None
            if message.get("type") == "audio_chunk":
                # Handle continuous audio chunks for VAD
                try:
                    audio_data = base64.b64decode(message["audio"])
                    client_data = client_vad_handlers.get(client_id)

                    if client_data:
                        try:
                            # Convert raw audio bytes to numpy array
                            # The frontend sends Int16Array as bytes, so we need to convert it back
                            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                            # Convert to float32 and normalize to [-1, 1]
                            audio_array = audio_int16.astype(np.float32) / 32768.0

                            # Process audio chunks
                            chunks = client_data['processor'].add_audio(audio_array)

                            for chunk in chunks:
                                is_speaking, complete_speech = client_data['vad'].process_audio_chunk(
                                    chunk, client_data['current_time']
                                )
                                client_data['current_time'] += 0.1  # 100ms chunks

                                if complete_speech is not None:
                                    # Complete speech detected, send clear signal and process it
                                    await websocket.send_text(json.dumps({
                                        "type": "clear_processing_log"
                                    }))
                                    await process_complete_speech(websocket, complete_speech, step_log=[])

                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}")
                            # Log more details for debugging
                            logger.error(f"Audio data length: {len(audio_data)}, type: {type(audio_data)}")

                except Exception as e:
                    logger.error(f"Error handling audio chunk: {e}")

            elif message.get("type") == "audio":
                try:
                    audio_data = base64.b64decode(message["audio"])
                    step_log.append("Received audio from user.")
                    nepali_text = stt_handler.transcribe(audio_data)
                    step_log.append(f"STT output: {nepali_text}")

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

                    agent_decision = agent.classify(nepali_text)
                    step_log.append(f"Agent decision: {agent_decision}")

                    if agent_decision == "rag":
                        # Direct Nepali query to RAG (no translation needed)
                        nepali_response = rag_handler.get_nepali_answer(nepali_text)
                        step_log.append(f"RAG Nepali response: {nepali_response}")
                    elif agent_decision == "goodbye":
                        nepali_response = agent.goodbye_conversation(nepali_text)
                        step_log.append(f"Goodbye response: {nepali_response}")
                    else:
                        nepali_response = agent.normal_conversation(nepali_text)
                        step_log.append(f"LLM conversational response: {nepali_response}")
                    audio_response = tts_handler.synthesize(nepali_response)
                    step_log.append("TTS synthesized audio.")
                    response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "agent_decision": agent_decision,
                        "rag_context": rag_context,
                        "step_log": step_log
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": response_audio_b64
                    }))
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}",
                        "step_log": step_log
                    }))
            elif message.get("type") == "text":
                try:
                    input_text = message.get("text", "")
                    step_log.append(f"Received text: {input_text}")

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

                    agent_decision = agent.classify(input_text)
                    step_log.append(f"Agent decision: {agent_decision}")

                    if agent_decision == "rag":
                        # Direct Nepali query to RAG (no translation needed)
                        nepali_response = rag_handler.get_nepali_answer(input_text)
                        step_log.append(f"RAG Nepali response: {nepali_response}")
                    elif agent_decision == "goodbye":
                        nepali_response = agent.goodbye_conversation(input_text)
                        step_log.append(f"Goodbye response: {nepali_response}")
                    else:
                        nepali_response = agent.normal_conversation(input_text)
                        step_log.append(f"LLM conversational response: {nepali_response}")
                    audio_response = tts_handler.synthesize(nepali_response)
                    step_log.append("TTS synthesized audio.")
                    response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "agent_decision": agent_decision,
                        "rag_context": rag_context,
                        "step_log": step_log
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": response_audio_b64
                    }))
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Text processing error: {str(e)}",
                        "step_log": step_log
                    }))
            elif message.get("type") == "vad_sensitivity":
                # Adjust VAD sensitivity
                try:
                    sensitivity = float(message.get("sensitivity", 0.5))
                    client_data = client_vad_handlers.get(client_id)
                    if client_data:
                        client_data['vad'].set_sensitivity(sensitivity)
                        await websocket.send_text(json.dumps({
                            "type": "vad_sensitivity_updated",
                            "sensitivity": sensitivity
                        }))
                except Exception as e:
                    logger.error(f"Error setting VAD sensitivity: {e}")

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
        # Clean up client VAD handler
        if client_id in client_vad_handlers:
            del client_vad_handlers[client_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Clean up client VAD handler
        if client_id in client_vad_handlers:
            del client_vad_handlers[client_id]
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass

async def process_complete_speech(websocket: WebSocket, audio_data: np.ndarray, step_log: list):
    """Process a complete speech segment detected by VAD with real-time updates"""
    import time

    try:
        # Convert numpy array to audio bytes for STT
        start_time = time.time()

        # Send real-time update: Starting STT
        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "stt",
            "status": "processing",
            "message": "🎤 Speech-to-Text प्रक्रिया सुरु भयो...",
            "timestamp": time.time()
        }))

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_data, 16000, format='WAV')
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()

        step_log.append("VAD detected complete speech segment.")

        # Process STT
        stt_start = time.time()
        nepali_text = stt_handler.transcribe(audio_bytes)
        stt_time = time.time() - stt_start

        step_log.append(f"STT output: {nepali_text}")

        # Send STT completion update
        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "stt",
            "status": "completed",
            "message": f"✅ Speech-to-Text सम्पन्न ({stt_time:.1f}s)",
            "result": nepali_text,
            "duration": stt_time,
            "timestamp": time.time()
        }))

        # Agent decision
        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "agent",
            "status": "processing",
            "message": "🤖 Agent निर्णय गर्दै...",
            "timestamp": time.time()
        }))

        agent_start = time.time()
        agent_decision = agent.classify(nepali_text)
        agent_time = time.time() - agent_start

        step_log.append(f"Agent decision: {agent_decision}")

        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "agent",
            "status": "completed",
            "message": f"✅ Agent निर्णय: {agent_decision} ({agent_time:.1f}s)",
            "result": agent_decision,
            "duration": agent_time,
            "timestamp": time.time()
        }))

        # Always send initial wait message for ALL queries to maintain consistent perceived latency
        buffer_message = "तपाईंको प्रश्नको लागि धन्यवाद। जवाफ तयार पार्दै छु..."
        await websocket.send_text(json.dumps({
            "type": "text_response",
            "text": buffer_message,
            "agent_decision": "searching",
            "step_log": step_log
        }))
        # Also send buffer message as audio for immediate playback
        buffer_audio = tts_handler.synthesize(buffer_message)
        buffer_audio_b64 = base64.b64encode(buffer_audio).decode('utf-8')
        await websocket.send_text(json.dumps({
            "type": "audio_response",
            "audio": buffer_audio_b64,
            "is_buffer": True
        }))

        rag_context = None
        if agent_decision == "rag":

            # Direct RAG processing with multilingual embeddings
            await websocket.send_text(json.dumps({
                "type": "realtime_update",
                "task": "rag",
                "status": "processing",
                "message": "🔍 Knowledge Base मा खोज्दै...",
                "timestamp": time.time()
            }))

            rag_start = time.time()
            nepali_response = rag_handler.get_nepali_answer(nepali_text)
            rag_time = time.time() - rag_start

            step_log.append(f"RAG Nepali response: {nepali_response}")

            await websocket.send_text(json.dumps({
                "type": "realtime_update",
                "task": "rag",
                "status": "completed",
                "message": f"✅ जवाफ तयार ({rag_time:.1f}s)",
                "duration": rag_time,
                "timestamp": time.time()
            }))

        else:
            # Normal conversation LLM
            await websocket.send_text(json.dumps({
                "type": "realtime_update",
                "task": "llm",
                "status": "processing",
                "message": "🧠 AI मोडेलले जवाफ तयार गर्दै...",
                "timestamp": time.time()
            }))

            llm_start = time.time()
            nepali_response = agent.normal_conversation(nepali_text)
            llm_time = time.time() - llm_start

            step_log.append(f"LLM conversational response: {nepali_response}")

            await websocket.send_text(json.dumps({
                "type": "realtime_update",
                "task": "llm",
                "status": "completed",
                "message": f"✅ AI जवाफ तयार ({llm_time:.1f}s)",
                "duration": llm_time,
                "timestamp": time.time()
            }))

        # TTS processing step
        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "tts",
            "status": "processing",
            "message": "🔊 Text-to-Speech तयार गर्दै...",
            "timestamp": time.time()
        }))

        tts_start = time.time()
        audio_response = tts_handler.synthesize(nepali_response)
        tts_time = time.time() - tts_start

        step_log.append("TTS synthesized audio.")
        response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')

        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "tts",
            "status": "completed",
            "message": f"✅ Audio तयार ({tts_time:.1f}s)",
            "duration": tts_time,
            "timestamp": time.time()
        }))

        # Calculate total processing time
        total_time = time.time() - start_time

        # Send final completion update
        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "complete",
            "status": "completed",
            "message": f"🎉 सम्पूर्ण प्रक्रिया सम्पन्न! कुल समय: {total_time:.1f}s",
            "total_duration": total_time,
            "timestamp": time.time()
        }))

        # Send final response
        await websocket.send_text(json.dumps({
            "type": "text_response",
            "text": nepali_response,
            "agent_decision": agent_decision,
            "rag_context": rag_context,
            "step_log": step_log,
            "input_text": nepali_text,
            "processing_time": total_time
        }))

        await websocket.send_text(json.dumps({
            "type": "audio_response",
            "audio": response_audio_b64
        }))

    except Exception as e:
        logger.error(f"Error processing complete speech: {e}")
        await websocket.send_text(json.dumps({
            "type": "realtime_update",
            "task": "error",
            "status": "error",
            "message": f"❌ Error: {str(e)}",
            "timestamp": time.time()
        }))
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Speech processing error: {str(e)}",
            "step_log": step_log
        }))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Voice Call Bot FastAPI Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 