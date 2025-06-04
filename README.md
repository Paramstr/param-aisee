# Osmo Assistant MVP

An always-listening, always-watching AI assistant running locally on macOS with M1 Pro. Features real-time audio processing, computer vision, and LLM integration with a modern web dashboard.

## Features

- 🎤 **Wake Word Detection**: Uses Porcupine for "Osmo" wake word detection
- 🔊 **Speech Recognition**: Whisper-based transcription with voice activity detection
- 📷 **Computer Vision**: Live camera feed with OpenCV integration
- 🤖 **LLM Integration**: OpenRouter API with vision-language model support
- 🗣️ **Text-to-Speech**: macOS native `say` command integration
- 🌐 **Real-time Dashboard**: Next.js frontend with WebSocket communication
- ⚡ **Event-Driven Architecture**: Async event bus for component coordination

## Architecture

```
osmo-assistant/
├── backend/                 # FastAPI Python backend
│   ├── app.py              # Main FastAPI application
│   ├── config.py           # Pydantic settings
│   ├── events.py           # Event system
│   └── core/
│       ├── audio.py        # Audio processing (Porcupine + Whisper)
│       ├── vision.py       # Camera capture (OpenCV)
│       ├── llm.py          # LLM client (OpenRouter)
│       └── tasks.py        # Task coordination
└── frontend/               # Next.js React frontend
    └── src/
        ├── lib/useSocket.ts    # WebSocket hook
        ├── components/         # UI components
        └── app/page.tsx        # Main dashboard
```

## Prerequisites

- macOS (tested on M1 Pro)
- Python 3.11+
- Node.js 18+
- Camera and microphone access
- OpenRouter API key
- Porcupine access key

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd osmo-assistant

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install "fastapi==0.111.*" "uvicorn[standard]==0.30.*" aiohttp \
           sounddevice webrtcvad "pvporcupine==3.*" whispercpp \
           pydantic opencv-python-headless pillow python-dotenv
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=qwen/qwen2.5-vl-72b-instruct:free

# Porcupine Wake Word Configuration
PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here
PORCUPINE_SENSITIVITY=0.5

# Audio Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_SIZE=1024
VAD_AGGRESSIVENESS=3
SILENCE_DURATION_THRESHOLD=2.0

# Vision Configuration
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# LLM Configuration
MAX_TOKENS=1000
TEMPERATURE=0.7

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### 3. Setup Frontend

```bash
# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 4. Get API Keys

#### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and get your API key
3. Add to `.env` file



## Running the Application

### Start Backend (Terminal 1)
```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI server
uvicorn backend.app:app --reload --port 8000
```

### Start Frontend (Terminal 2)
```bash
# Start Next.js development server
cd frontend
npm run dev
```

### Access Dashboard
Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. **System Startup**: Both backend and frontend should be running
2. **Camera Permission**: Grant camera access when prompted
3. **Microphone Permission**: Grant microphone access when prompted
4. **Wake Word**: Say "Osmo" (or "computer" as fallback) to activate
5. **Voice Command**: Speak your question after the wake word
6. **AI Response**: Watch the live transcription and AI response
7. **TTS Playback**: Listen to the AI response via macOS speech synthesis

## API Endpoints

- `GET /health` - Health check
- `GET /frame` - Current camera frame (JPEG)
- `GET /status` - System status
- `WebSocket /ws` - Real-time event stream
- `POST /test-wake-word` - Test wake word detection
- `POST /test-transcript` - Test transcript processing

## Event Types

- `wake_word_detected` - Wake word triggered
- `audio_recording_start/stop` - Recording state changes
- `transcript_ready` - Speech-to-text complete
- `llm_response_start/chunk/end` - LLM processing states
- `tts_start/end` - Text-to-speech states
- `frame_captured` - Camera frame events
- `status_update` - System status changes
- `error` - Error events

## Troubleshooting

### Audio Issues
- Check microphone permissions in System Preferences
- Verify audio device selection
- Test with different `CAMERA_INDEX` values

### Camera Issues
- Check camera permissions in System Preferences
- Verify camera is not in use by other applications
- Test with different `CAMERA_INDEX` values

### API Issues
- Verify OpenRouter API key is valid
- Check internet connection
- Monitor backend logs for API errors

### Wake Word Issues
- Verify Porcupine access key is valid
- Adjust `PORCUPINE_SENSITIVITY` (0.0-1.0)
- Try the fallback "computer" keyword

## Development

### Backend Development
```bash
# Run with auto-reload
uvicorn backend.app:app --reload --port 8000

# Run tests (if implemented)
pytest backend/tests/
```

### Frontend Development
```bash
# Run development server
cd frontend
npm run dev

# Build for production
npm run build
```

### Adding Custom Wake Words
1. Create custom keyword file using Picovoice Console
2. Download `.ppn` file
3. Set `PORCUPINE_KEYWORD_PATH` in `.env`

## Performance Notes

- **Whisper Model**: Uses "tiny" model for speed (can be upgraded)
- **Camera FPS**: Default 30fps (adjustable via `CAMERA_FPS`)
- **LLM Streaming**: Real-time token streaming for responsiveness
- **Event Queue**: Bounded queues prevent memory issues

## Security Considerations

- API keys stored in `.env` (add to `.gitignore`)
- Local processing for privacy
- CORS configured for localhost only
- No external data transmission except LLM API calls

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 