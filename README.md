# 🎙️ ShikshaSathi - AI Voice Assistant for Kathmandu University

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**ShikshaSathi** is an intelligent voice-to-voice AI assistant designed specifically for Kathmandu University School of Engineering (KUSOE). It provides real-time voice interactions in Nepali and English, helping prospective students with admissions, course information, fees, and academic queries.

## ✨ Features

### 🗣️ **Voice Interaction**
- **Real-time Speech Recognition** - Converts speech to text using advanced STT models
- **Natural Voice Synthesis** - High-quality TTS with Nepali language support
- **Voice Activity Detection** - Smart detection with noise filtering and background noise adaptation
- **Multilingual Support** - Seamless handling of Nepali, English, and mixed-language queries

### 🧠 **Intelligent Response System**
- **RAG-Powered Knowledge Base** - Retrieval-Augmented Generation with university-specific data
- **Smart Query Classification** - Automatically routes queries to appropriate response systems
- **Context-Aware Responses** - Maintains conversation context and provides relevant information
- **Fallback to Human Support** - Graceful handoff to customer representatives for complex queries

### 🎯 **University-Specific Features**
- **Admission Information** - Complete details about KUCAT entrance exams and admission process
- **Course Details** - Comprehensive information about all engineering programs
- **Fee Structure** - Detailed breakdown of fees for all programs
- **Academic Calendar** - Important dates and deadlines
- **Scholarship Information** - Available scholarships and eligibility criteria

### 🔧 **Technical Features**
- **WebSocket Communication** - Real-time bidirectional communication
- **Adaptive Noise Filtering** - Dynamic background noise suppression
- **Quality-Based Speech Detection** - Filters out partial words and noise
- **Responsive Web Interface** - Modern, mobile-friendly design
- **Scalable Architecture** - Modular design for easy maintenance and updates

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │  AI Services    │
│                 │    │                 │    │                 │
│ • Web Interface │◄──►│ • FastAPI       │◄──►│ • Gemini LLM    │
│ • WebSocket     │    │ • WebSocket     │    │ • ChromaDB      │
│ • Audio Player  │    │ • VAD Handler   │    │ • Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Knowledge Base  │
                    │                 │
                    │ • KUSOE Data    │
                    │ • Vector Store  │
                    │ • RAG Pipeline  │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Microphone and speakers/headphones
- Modern web browser with WebRTC support

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/amanjha10/ShikshaSathi.git
cd ShikshaSathi
```

2. **Create virtual environment**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

5. **Initialize the knowledge base**
```bash
python rebuild_vector_db.py
```

6. **Start the server**
```bash
cd app/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

7. **Open your browser**
Navigate to `http://localhost:8000` and start talking!

## 📁 Project Structure

```
ShikshaSathi/
├── app/
│   ├── backend/           # FastAPI backend server
│   │   ├── main.py       # Main server application
│   │   ├── vad_handler.py # Voice Activity Detection
│   │   ├── stt_handler.py # Speech-to-Text processing
│   │   ├── tts_handler.py # Text-to-Speech synthesis
│   │   ├── rag_handler.py # RAG system implementation
│   │   └── agent.py      # Query classification and routing
│   └── frontend/         # Web interface
│       ├── index.html    # Main HTML page
│       ├── app.js        # Frontend JavaScript
│       └── style.css     # Styling
├── pipeline/             # Alternative pipeline implementation
├── KUSOE_database/       # University knowledge base
│   ├── overview.txt      # General university information
│   └── programs/         # Program-specific details
├── vector-db/            # ChromaDB vector database
├── requirements.txt      # Python dependencies
├── rebuild_vector_db.py  # Database initialization script
└── README.md            # This file
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
CUDA_VISIBLE_DEVICES=0
```

### VAD Settings
Adjust voice detection sensitivity in `app/backend/vad_handler.py`:

```python
# More sensitive detection
vad = VADHandler(
    energy_threshold=0.015,
    silence_duration=1.5,
    min_speech_duration=0.6
)
```

## 🎯 Usage Examples

### Basic Voice Interaction
1. Open the web interface
2. Click the microphone button or just start speaking
3. Ask questions like:
   - "Computer engineering ko fee kati ho?" (What's the computer engineering fee?)
   - "KUSOE ma admission process ke ho?" (What's the admission process at KUSOE?)
   - "कृत्रिम बुद्धिमत्ता कोर्स छ कि छैन?" (Is there an AI course available?)

### Text Input
- Type your questions directly in the text input field
- Supports both Nepali and English text

### Voice Commands
- **"नमस्ते"** - Greeting and introduction
- **"धन्यवाद"** - Thank you and goodbye
- **"तपाईं को हुनुहुन्छ?"** - Ask about the assistant

## 🔧 Development

### Adding New Knowledge
1. Add content to `KUSOE_database/` directory
2. Run the rebuild script:
```bash
python rebuild_vector_db.py
```

### Customizing Responses
Edit the prompt templates in `app/backend/rag_handler.py` and `app/backend/agent.py`

### Adjusting Voice Detection
Modify parameters in `app/backend/vad_handler.py` for different environments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

## 📊 Performance

- **Response Time**: < 3 seconds for most queries
- **Voice Detection**: Adaptive noise filtering with 95%+ accuracy
- **Knowledge Base**: 60+ documents with multilingual embeddings
- **Concurrent Users**: Supports multiple simultaneous connections

## 🛠️ Troubleshooting

### Common Issues

**Voice not detected:**
- Check microphone permissions
- Adjust VAD sensitivity settings
- Ensure quiet environment for initial setup

**Slow responses:**
- Verify GEMINI_API_KEY is set correctly
- Check internet connection
- Consider using GPU acceleration

**Knowledge base errors:**
- Run `python rebuild_vector_db.py`
- Check ChromaDB installation
- Verify KUSOE_database content

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
