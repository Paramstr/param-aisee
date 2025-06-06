#!/usr/bin/env python3
"""
Test script for video recording functionality
Tests the VideoRecorder class and related video processing logic
"""

import asyncio
import logging
import sys
import os
import tempfile
import base64
import cv2
import time
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from backend.core.video_capture import VideoRecorder
    from backend.core.vision import VisionProcessor
    from backend.config import settings
    from backend.events import event_bus, Event, EventType
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestVisionProcessor:
    """Mock vision processor for testing"""
    
    def __init__(self):
        self.is_capturing = True
        self.frame_count = 0
        
    def get_latest_frame(self):
        """Generate a test frame with a counter"""
        # Create a simple test frame (640x480, blue background with counter text)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 100  # Blue channel
        
        # Add frame counter text
        cv2.putText(frame, f"Frame {self.frame_count}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        self.frame_count += 1
        return frame


async def test_basic_video_creation():
    """Test basic video file creation without the full recorder"""
    logger.info("🧪 Test: Basic video file creation")
    
    try:
        # Create a simple test video using OpenCV directly
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        video_path = temp_file.name
        
        # Video settings
        fps = 30
        duration = 2  # seconds
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video writer
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            logger.error("❌ Failed to initialize video writer")
            return False
        
        # Generate and write frames
        total_frames = fps * duration
        for i in range(total_frames):
            # Create a test frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 1] = 150  # Green channel
            
            # Add frame number
            cv2.putText(frame, f"Frame {i+1}/{total_frames}", (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            video_writer.write(frame)
        
        video_writer.release()
        
        # Verify the created video
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            logger.info(f"✅ Video file created: {os.path.getsize(video_path)} bytes")
            
            # Test reading the video back
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"✅ Video verification: {frame_count} frames at {video_fps} fps")
                cap.release()
            else:
                logger.error("❌ Could not read back the created video")
                
            # Convert to base64 for size test
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                logger.info(f"✅ Base64 encoding: {len(video_base64)} characters")
            
            # Cleanup
            os.unlink(video_path)
            return True
        else:
            logger.error("❌ Video file was not created or is empty")
            return False
            
    except Exception as e:
        logger.error(f"❌ Basic video creation test failed: {e}")
        return False


async def test_video_recording():
    """Test the video recording functionality"""
    logger.info("🧪 Starting video recording tests...")
    
    # Create mock vision processor
    vision_processor = TestVisionProcessor()
    
    # Create video recorder
    video_recorder = VideoRecorder(vision_processor)
    
    # Test 1: Basic video recording
    logger.info("🧪 Test 1: Basic 3-second video recording")
    result = await video_recorder.record_video(3)
    
    if result["success"]:
        logger.info(f"✅ Video recording successful!")
        logger.info(f"   Duration: {result['duration']}s")
        logger.info(f"   File size: {result['file_size']} bytes")
        logger.info(f"   Frames recorded: {result['frames_recorded']}")
        logger.info(f"   Base64 length: {len(result['video_base64'])} chars")
        
        # Test 2: Decode and save video for inspection
        logger.info("🧪 Test 2: Decode and save video file")
        try:
            video_bytes = base64.b64decode(result['video_base64'])
            
            # Save to temporary file for inspection
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name
            
            logger.info(f"✅ Video saved to: {temp_path}")
            
            # Test 3: Verify video properties using OpenCV
            logger.info("🧪 Test 3: Verify video properties")
            cap = cv2.VideoCapture(temp_path)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                logger.info(f"✅ Video properties verified:")
                logger.info(f"   FPS: {fps}")
                logger.info(f"   Frame count: {frame_count}")
                logger.info(f"   Dimensions: {width}x{height}")
                logger.info(f"   Calculated duration: {duration:.2f}s")
                
                # Test 4: Read and verify some frames
                logger.info("🧪 Test 4: Verify frame content")
                frame_samples = []
                for i in range(min(5, frame_count)):  # Sample first 5 frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * (frame_count // 5) if frame_count > 5 else i)
                    ret, frame = cap.read()
                    if ret:
                        frame_samples.append((i, frame.shape, frame.dtype))
                
                cap.release()
                
                for i, shape, dtype in frame_samples:
                    logger.info(f"   Frame {i}: shape={shape}, dtype={dtype}")
                
                logger.info("✅ Frame verification completed")
                
                # Test 5: Performance metrics
                logger.info("🧪 Test 5: Performance analysis")
                expected_frames = result['duration'] * 30  # 30 FPS target
                frame_efficiency = (result['frames_recorded'] / expected_frames) * 100
                
                logger.info(f"   Expected frames (30fps): {expected_frames}")
                logger.info(f"   Actual frames: {result['frames_recorded']}")
                logger.info(f"   Frame efficiency: {frame_efficiency:.1f}%")
                
                if frame_efficiency >= 90:
                    logger.info("✅ Frame rate performance is good")
                else:
                    logger.warning("⚠️ Frame rate performance is below 90%")
            
            else:
                logger.error("❌ Could not open saved video file for verification")
            
            # Cleanup
            try:
                os.unlink(temp_path)
                logger.info("🧹 Cleanup: Temporary video file deleted")
            except:
                logger.warning(f"⚠️ Could not delete temporary file: {temp_path}")
                
        except Exception as e:
            logger.error(f"❌ Video decoding test failed: {e}")
    
    else:
        logger.error(f"❌ Video recording failed: {result.get('error')}")
        return False
    
    # Test 6: Test recording while already recording (should fail)
    logger.info("🧪 Test 6: Concurrent recording test")
    
    # Start a long recording
    long_recording_task = asyncio.create_task(video_recorder.record_video(5))
    
    # Wait a bit then try to start another
    await asyncio.sleep(0.5)
    concurrent_result = await video_recorder.record_video(2)
    
    if not concurrent_result["success"] and "Already recording" in concurrent_result.get("error", ""):
        logger.info("✅ Concurrent recording properly rejected")
    else:
        logger.warning("⚠️ Concurrent recording should have been rejected")
    
    # Wait for long recording to complete
    long_result = await long_recording_task
    if long_result["success"]:
        logger.info("✅ Long recording completed successfully")
    
    # Test 7: Test availability check
    logger.info("🧪 Test 7: Availability check")
    if video_recorder.is_available():
        logger.info("✅ Video recorder reports as available")
    else:
        logger.warning("⚠️ Video recorder reports as unavailable")
    
    logger.info("🧪 All video recording tests completed!")
    return True


async def test_vision_processor_integration():
    """Test with actual vision processor if camera is available"""
    logger.info("🧪 Testing with actual vision processor...")
    
    try:
        # Try to create actual vision processor
        vision_processor = VisionProcessor()
        await vision_processor.start()
        
        if vision_processor.is_capturing:
            logger.info("✅ Camera is available, testing with real frames")
            
            video_recorder = VideoRecorder(vision_processor)
            result = await video_recorder.record_video(2)
            
            if result["success"]:
                logger.info("✅ Real camera recording successful!")
                logger.info(f"   File size: {result['file_size']} bytes")
                logger.info(f"   Frames: {result['frames_recorded']}")
            else:
                logger.error(f"❌ Real camera recording failed: {result.get('error')}")
            
            await vision_processor.stop()
        else:
            logger.warning("⚠️ No camera available, skipping real camera test")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not test with real camera: {e}")


async def main():
    """Main test runner"""
    logger.info("🎬 Video Recording Test Suite")
    logger.info("=" * 50)
    
    try:
        # Test 1: Basic video creation (without dependencies)
        basic_success = await test_basic_video_creation()
        
        if basic_success:
            logger.info("✅ Basic video creation works")
            
            # Test 2: Full video recorder functionality
            success = await test_video_recording()
            
            if success:
                logger.info("✅ Video recorder functionality works")
            else:
                logger.error("❌ Video recorder tests failed")
        else:
            logger.error("❌ Basic video creation failed - check OpenCV installation")
        
        logger.info("=" * 50)
        logger.info("🧪 Test suite completed!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main()) 