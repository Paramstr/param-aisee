# Bus Demo Setup

This demo compares **Moondream Cloud** vs **Local Moondream Server** for bus number detection with real-time latency measurements.

## 🚀 Quick Setup

### Option 1: Moondream Cloud (Easiest)
1. Get a free API key from [Moondream Cloud Console](https://moondream.ai) (5,000 free requests/day)
2. Add to your `.env` file:
   ```
   MOONDREAM_API_KEY=your_api_key_here
   ```

### Option 2: Local Moondream Server  
1. Install Moondream server:
   ```bash
   pip install moondream
   moondream serve
   ```
2. Server runs at `http://localhost:2020/v1` by default

## 📹 Video Setup

Place your MP4 bus videos using this exact naming:
- `bus_video_1.mp4` → Shows as "Bus Video 1"  
- `bus_video_2.mp4` → Shows as "Bus Video 2"
- `bus_video_3.mp4` → Shows as "Bus Video 3"
- etc. (supports up to 5 videos)

## 🎯 How It Works

1. **Select Video**: Choose from available bus videos
2. **Toggle Mode**: Switch between ☁️ Cloud and 🖥️ Local inference  
3. **Start Detection**: Process frames with real-time results
4. **Compare Performance**: See latency differences between cloud vs local

## 📊 Features

- **Real-time Processing**: 1 frame per second analysis
- **Live Results**: WebSocket streaming of detections
- **Performance Metrics**: Latency measurements and detection counts
- **Mode Switching**: Toggle between cloud/local during runtime
- **Frame Preview**: See exact frames where buses were detected

## 🔧 API Endpoints

- `GET /object-demo/videos` - List available videos
- `POST /object-demo/start/{video_id}` - Start detection  
- `POST /object-demo/stop` - Stop detection
- `POST /object-demo/inference-mode` - Switch cloud/local
- `GET /object-demo/status` - Check availability

## 💡 Tips

- **Cloud**: Higher accuracy, network latency, API limits
- **Local**: Lower latency, requires local setup, unlimited usage
- **Videos**: Use clear footage with visible bus numbers for best results 