#!/usr/bin/env python3
"""
Focused test script to identify potential video recording issues
Tests edge cases and common problems that could cause video recording failures
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

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from backend.core.video_capture import VideoRecorder
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IssueTestVisionProcessor:
    """Mock vision processor to test edge cases"""
    
    def __init__(self, test_type="normal"):
        self.is_capturing = True
        self.frame_count = 0
        self.test_type = test_type
        
    def get_latest_frame(self):
        """Generate test frames based on test type"""
        self.frame_count += 1
        
        if self.test_type == "normal":
            # Normal 640x480 frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 2] = 200  # Red channel
            cv2.putText(frame, f"Normal {self.frame_count}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif self.test_type == "variable_size":
            # Variable frame sizes - this should cause issues
            sizes = [(640, 480), (320, 240), (800, 600)]
            w, h = sizes[self.frame_count % len(sizes)]
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 1] = 150  # Green channel
            cv2.putText(frame, f"Var {w}x{h}", (50, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif self.test_type == "wrong_dtype":
            # Wrong data type (float instead of uint8)
            frame = np.zeros((480, 640, 3), dtype=np.float32)
            frame[:, :, 0] = 0.8  # Blue channel (0-1 range)
            # Note: can't add text to float32 frame easily
            
        elif self.test_type == "none_frames":
            # Intermittently return None
            if self.frame_count % 3 == 0:
                return None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 1] = 100
            cv2.putText(frame, f"Some {self.frame_count}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif self.test_type == "slow_frames":
            # Simulate slow frame generation
            time.sleep(0.1)  # 100ms delay
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = 180  # Blue
            cv2.putText(frame, f"Slow {self.frame_count}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Default case
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        return frame


async def test_codec_availability():
    """Test which video codecs are available"""
    logger.info("🧪 Testing codec availability...")
    
    codecs_to_test = [
        ('mp4v', 'MPEG-4 Part 2'),
        ('XVID', 'Xvid MPEG-4'),
        ('H264', 'H.264'),
        ('avc1', 'H.264 (Apple)'),
        ('MJPG', 'Motion JPEG'),
        ('DIVX', 'DivX')
    ]
    
    available_codecs = []
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()
    temp_path = temp_file.name
    
    for codec_name, description in codecs_to_test:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            writer = cv2.VideoWriter(temp_path, fourcc, 30, (640, 480))
            
            if writer.isOpened():
                logger.info(f"✅ {codec_name} ({description}) - Available")
                available_codecs.append((codec_name, description))
                writer.release()
            else:
                logger.warning(f"⚠️ {codec_name} ({description}) - Not available")
                
        except Exception as e:
            logger.warning(f"❌ {codec_name} ({description}) - Error: {e}")
    
    os.unlink(temp_path)
    logger.info(f"📊 Available codecs: {len(available_codecs)}/{len(codecs_to_test)}")
    return available_codecs


async def test_frame_size_consistency():
    """Test handling of inconsistent frame sizes"""
    logger.info("🧪 Testing frame size consistency...")
    
    vision_processor = IssueTestVisionProcessor("variable_size")
    video_recorder = VideoRecorder(vision_processor)
    
    result = await video_recorder.record_video(2)
    
    if result["success"]:
        logger.info("✅ Variable frame sizes handled successfully")
        logger.info(f"   File size: {result['file_size']} bytes")
        logger.info(f"   Frames: {result['frames_recorded']}")
    else:
        logger.warning(f"⚠️ Variable frame sizes caused issues: {result.get('error')}")
    
    return result["success"]


async def test_frame_data_types():
    """Test handling of wrong frame data types"""
    logger.info("🧪 Testing frame data type handling...")
    
    vision_processor = IssueTestVisionProcessor("wrong_dtype")
    video_recorder = VideoRecorder(vision_processor)
    
    result = await video_recorder.record_video(2)
    
    if result["success"]:
        logger.info("✅ Wrong data types handled successfully")
        logger.info(f"   File size: {result['file_size']} bytes")
        logger.info(f"   Frames: {result['frames_recorded']}")
    else:
        logger.warning(f"⚠️ Wrong data types caused issues: {result.get('error')}")
    
    return result["success"]


async def test_none_frames():
    """Test handling of None frames"""
    logger.info("🧪 Testing None frame handling...")
    
    vision_processor = IssueTestVisionProcessor("none_frames")
    video_recorder = VideoRecorder(vision_processor)
    
    result = await video_recorder.record_video(2)
    
    if result["success"]:
        logger.info("✅ None frames handled successfully")
        logger.info(f"   File size: {result['file_size']} bytes")
        logger.info(f"   Frames: {result['frames_recorded']}")
    else:
        logger.warning(f"⚠️ None frames caused issues: {result.get('error')}")
    
    return result["success"]


async def test_slow_frame_generation():
    """Test handling of slow frame generation"""
    logger.info("🧪 Testing slow frame generation...")
    
    vision_processor = IssueTestVisionProcessor("slow_frames")
    video_recorder = VideoRecorder(vision_processor)
    
    start_time = time.time()
    result = await video_recorder.record_video(2)
    duration = time.time() - start_time
    
    if result["success"]:
        logger.info("✅ Slow frames handled successfully")
        logger.info(f"   Actual recording time: {duration:.2f}s")
        logger.info(f"   File size: {result['file_size']} bytes")
        logger.info(f"   Frames: {result['frames_recorded']}")
        
        if duration > 4:  # Should be ~2s but slow frames add delay
            logger.info("✅ Recording properly waited for slow frames")
        else:
            logger.warning("⚠️ Recording might not have waited for all frames")
    else:
        logger.warning(f"⚠️ Slow frames caused issues: {result.get('error')}")
    
    return result["success"]


async def test_concurrent_access():
    """Test the concurrent recording protection more thoroughly"""
    logger.info("🧪 Testing concurrent access protection...")
    
    vision_processor = IssueTestVisionProcessor("normal")
    video_recorder = VideoRecorder(vision_processor)
    
    # Start multiple recordings simultaneously
    tasks = []
    for i in range(3):
        task = asyncio.create_task(video_recorder.record_video(2))
        tasks.append(task)
        await asyncio.sleep(0.1)  # Small delay between starts
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_recordings = 0
    failed_recordings = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Recording {i+1}: Exception - {result}")
            failed_recordings += 1
        elif isinstance(result, dict):
            if result.get("success"):
                logger.info(f"Recording {i+1}: Success - {result['frames_recorded']} frames")
                successful_recordings += 1
            else:
                logger.info(f"Recording {i+1}: Failed - {result.get('error')}")
                failed_recordings += 1
    
    logger.info(f"📊 Concurrent test results: {successful_recordings} successful, {failed_recordings} failed")
    
    # Should have exactly 1 successful recording
    if successful_recordings == 1 and failed_recordings == 2:
        logger.info("✅ Concurrent access properly protected")
        return True
    else:
        logger.warning("⚠️ Concurrent access protection may have issues")
        return False


async def main():
    """Main test runner for issue detection"""
    logger.info("🔍 Video Recording Issue Detection Suite")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Codec availability
        available_codecs = await test_codec_availability()
        test_results["codecs"] = len(available_codecs) > 0
        
        # Test 2: Frame size consistency
        test_results["frame_sizes"] = await test_frame_size_consistency()
        
        # Test 3: Data type handling
        test_results["data_types"] = await test_frame_data_types()
        
        # Test 4: None frame handling
        test_results["none_frames"] = await test_none_frames()
        
        # Test 5: Slow frame generation
        test_results["slow_frames"] = await test_slow_frame_generation()
        
        # Test 6: Concurrent access protection
        test_results["concurrent"] = await test_concurrent_access()
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 Test Results Summary:")
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"   {test_name}: {status}")
            if passed:
                passed_tests += 1
        
        logger.info(f"📈 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All issue detection tests passed!")
        else:
            logger.warning("⚠️ Some issues detected - review failed tests")
        
    except Exception as e:
        logger.error(f"❌ Issue detection suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 