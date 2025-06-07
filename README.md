# Osmo Assistant

An always-listening, always-watching AI assistant that runs locally on macOS. It combines real-time audio processing, computer vision, and large language models to create an interactive AI companion that can see and respond to its environment.

## Features

- 🎤 **Wake Word Detection**: Responds to "Osmo" wake word from transcription
- 🔊 **Speech Recognition**: MLX Whisper transcription with voice activity detection
- 📷 **Computer Vision**: Live camera feed with OpenCV integration  
- 🤖 **LLM Integration**: OpenRouter API with vision-language model support
- 🗣️ **Text-to-Speech**: macOS native `say` command integration
- 🌐 **Real-time Dashboard**: Next.js frontend with WebSocket communication
- ⚡ **Event-Driven Architecture**: Async event bus for component coordination
- 🛠️ **Tool System**: Photo capture and video recording capabilities
- 💾 **Conversation Memory**: Persistent conversation history with context management

## Architecture

```
osmo-assistant/
├── backend/                     # Python FastAPI backend
│   ├── app.py                  # Main FastAPI application & WebSocket server
│   ├── config.py               # Pydantic settings (env vars, API keys)
│   ├── events.py               # Event system (EventBus, EventType enum)
│   ├── requirements.txt        # Python dependencies
│   ├── recordings/             # Video recording storage
│   └── core/                   # Core processing modules
│       ├── audio.py           # Audio processing (MLX Whisper + VAD)
│       ├── vision.py          # Camera capture (OpenCV + JPEG encoding)
│       ├── video_capture.py   # Video recording system
│       ├── llm.py             # LLM client (OpenRouter API + TTS)
│       ├── tools.py           # Tool system (photo/video capture)
│       ├── conversation.py    # Conversation storage & context management
│       ├── shared.py          # Shared utilities and constants
│       └── tasks.py           # Task coordination & event handling
└── frontend/                   # Next.js React frontend
    └── src/
        ├── lib/useSocket.ts    # WebSocket hook for real-time events
        ├── components/         # React components
        └── app/page.tsx        # Main dashboard layout
```

## Prerequisites

- **macOS** (tested on M1 Pro, requires Apple Silicon for MLX Whisper)
- **Python 3.11+**
- **Node.js 18+** and **pnpm**
- **Camera and microphone access**
- **OpenRouter API key**

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd osmo-assistant

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenRouter API Configuration (Required)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=qwen/qwen2.5-vl-72b-instruct:free

# Audio Configuration (Optional - defaults provided)
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_SIZE=1024
VAD_AGGRESSIVENESS=3
SILENCE_DURATION_THRESHOLD=2.0

# Vision Configuration (Optional - defaults provided)
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30

# LLM Configuration (Optional - defaults provided)
MAX_TOKENS=1000
TEMPERATURE=0.7

# Server Configuration (Optional - defaults provided)
HOST=0.0.0.0
PORT=8000
```

### 3. Setup Frontend

```bash
# Install frontend dependencies using pnpm
cd frontend
pnpm install
cd ..
```

### 4. Get OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Generate your API key
4. Add it to your `.env` file as `OPENROUTER_API_KEY`

## Running the Application

### Option 1: Using Start Scripts (Recommended)

```bash
# Start backend (Terminal 1)
./start_backend.sh

# Start frontend (Terminal 2)  
./start_frontend.sh
```

### Option 2: Manual Start

```bash
# Terminal 1: Start Backend
source .venv/bin/activate
uvicorn backend.app:app --reload --port 8000

# Terminal 2: Start Frontend
cd frontend
pnpm dev
```

### 3. Access the Dashboard

Open [http://localhost:3000](http://localhost:3000) in your browser.

## How to Use

1. **Grant Permissions**: Allow camera and microphone access when prompted
2. **Wake Word**: Say "Osmo" to activate the assistant
3. **Voice Command**: Speak your question/request after activation
4. **Visual Context**: The assistant can see what your camera sees
5. **AI Response**: Watch live transcription and streaming AI responses
6. **Tool Usage**: Assistant can capture photos or record videos when requested
7. **Conversation History**: All interactions are saved and provide context

## Core Dependencies

### Backend (Python)
- **FastAPI 0.111.*** - Web framework with WebSocket support
- **MLX Whisper** - Speech transcription (Apple Silicon optimized)
- **OpenCV** - Computer vision and camera capture
- **WebRTC VAD** - Voice activity detection
- **OpenAI Client** - API integration with OpenRouter
- **Pydantic** - Configuration and data validation

### Frontend (Node.js)
- **Next.js 15.3.3** - React framework with app router
- **React 19.0.0** - UI library with latest features
- **TypeScript 5+** - Type safety and development experience
- **Tailwind CSS 4** - Modern styling framework

## API Endpoints

- `GET /health` - System health check
- `GET /frame` - Current camera frame (JPEG)
- `GET /status` - Component status overview
- `WebSocket /ws` - Real-time event stream
- `POST /test-transcript` - Test transcript processing

## Event System

The assistant uses an event-driven architecture with these event types:

- `wake_word_detected` - Wake word triggered from transcription
- `audio_recording_start/stop` - Recording state changes
- `transcript_ready` - Speech-to-text completion
- `llm_response_start/chunk/end` - LLM processing states
- `tool_event` - Tool execution status
- `tts_start/end` - Text-to-speech states
- `frame_captured` - Camera frame updates
- `status_update` - System status changes
- `error` - Error notifications

## Tool System

The assistant includes built-in tools:

- **Photo Capture**: `<get_photo/>` - Takes single still images
- **Video Recording**: `<get_video duration="N"/>` - Records 1-10 second clips
- **Tool Registry**: Manages available tools and execution
- **Event Integration**: Real-time tool status updates

## Troubleshooting

### Audio Issues
- **No transcription**: Check microphone permissions in System Preferences → Security & Privacy → Microphone
- **Poor recognition**: Adjust `VAD_AGGRESSIVENESS` (1-3) or `SILENCE_DURATION_THRESHOLD`
- **MLX errors**: Ensure you're on Apple Silicon Mac (M1/M2/M3)

### Camera Issues  
- **No video feed**: Check camera permissions in System Preferences → Security & Privacy → Camera
- **Wrong camera**: Try different `CAMERA_INDEX` values (0, 1, 2...)
- **Camera in use**: Close other applications using the camera

### API Issues
- **No LLM responses**: Verify `OPENROUTER_API_KEY` is valid and funded
- **Rate limits**: Check OpenRouter dashboard for usage limits
- **Network errors**: Ensure stable internet connection

### Wake Word Issues
- **Not detecting "Osmo"**: Speak clearly and ensure audio is being captured
- **False positives**: The system detects "Osmo" from transcription, so clear audio helps
- **Adjust sensitivity**: Modify `SILENCE_DURATION_THRESHOLD` for different speech patterns

## Development

### Backend Development
```bash
# Run with auto-reload
source .venv/bin/activate
uvicorn backend.app:app --reload --port 8000 --log-level debug

# Check logs for debugging
tail -f logs/app.log  # if logging to file
```

### Frontend Development
```bash
cd frontend
pnpm dev    # Development server with hot reload
pnpm build  # Production build
pnpm start  # Production server
```

### Adding Custom Tools
1. Define tool function in `backend/core/tools.py`
2. Register in tool registry
3. Add event handling in `backend/core/tasks.py`
4. Update frontend components if needed

## Performance Notes

- **Response Time**: <3 seconds from wake word to TTS output
- **MLX Whisper**: Optimized for Apple Silicon (much faster than CPU)
- **Streaming**: Real-time token streaming for immediate feedback
- **Memory**: Bounded event queues prevent memory leaks
- **Camera**: 30fps capture with efficient JPEG encoding

## Privacy & Security

- **Local Processing**: All audio/video processing happens locally
- **API Calls**: Only transcripts and images sent to OpenRouter
- **No Storage**: No persistent audio/video storage (except conversation history)
- **Permissions**: Granular camera/microphone permissions
- **Environment**: API keys stored in `.env` (add to `.gitignore`)

## System Requirements

- **macOS 12.0+** (for MLX compatibility)
- **Apple Silicon** (M1/M2/M3 for optimal MLX Whisper performance)
- **8GB+ RAM** (recommended for smooth operation)
- **Camera and Microphone** (built-in or external)
- **Internet Connection** (for OpenRouter API calls)

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For bug reports and feature requests, please use GitHub Issues. 