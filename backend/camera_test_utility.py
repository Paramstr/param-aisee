#!/usr/bin/env python3
"""
Camera Detection Utility for Osmo Assistant

Run this script to test your camera capabilities and find optimal settings.
This will help you configure the best camera settings for your system.

Usage:
    python camera_test_utility.py
"""

import cv2
import time
import sys

def test_camera_configurations():
    """Test various camera configurations to find what works best"""
    
    print("🔍 Osmo Assistant Camera Detection Utility")
    print("=" * 50)
    
    # Test configurations in order of preference
    test_configs = [
        (1920, 1080, 60, "4K-ready"),
        (1920, 1080, 30, "Full HD"),
        (1280, 720, 60, "HD 60fps"),
        (1280, 720, 30, "HD 30fps"),
        (640, 480, 30, "VGA (recommended)"),
        (640, 480, 15, "VGA low fps"),
        (320, 240, 30, "QVGA (basic)"),
    ]
    
    # Test camera indices
    working_cameras = []
    
    print("\n🎥 Detecting available cameras...")
    for camera_idx in range(5):  # Test first 5 camera indices
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                working_cameras.append((camera_idx, frame.shape))
                print(f"  ✅ Camera {camera_idx}: {frame.shape[1]}x{frame.shape[0]} (default)")
            cap.release()
        else:
            print(f"  ❌ Camera {camera_idx}: Not available")
    
    if not working_cameras:
        print("\n❌ No working cameras found!")
        return
    
    print(f"\n✅ Found {len(working_cameras)} working camera(s)")
    
    # Test configurations on the first working camera
    camera_idx = working_cameras[0][0]
    print(f"\n🧪 Testing configurations on camera {camera_idx}...")
    print("-" * 50)
    
    working_configs = []
    
    for width, height, fps, description in test_configs:
        print(f"\nTesting {description}: {width}x{height} @ {fps}fps")
        
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print("  ❌ Cannot open camera")
            continue
        
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Test frame capture
        success_count = 0
        total_tests = 5
        frame_times = []
        
        for i in range(total_tests):
            start_time = time.time()
            ret, frame = cap.read()
            frame_time = time.time() - start_time
            
            if ret:
                success_count += 1
                frame_times.append(frame_time)
            time.sleep(0.1)
        
        cap.release()
        
        if success_count == total_tests:
            avg_frame_time = sum(frame_times) / len(frame_times)
            actual_fps_measured = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            working_configs.append({
                'requested': (width, height, fps),
                'actual': (actual_width, actual_height, actual_fps),
                'description': description,
                'measured_fps': actual_fps_measured,
                'frame_time': avg_frame_time
            })
            
            print(f"  ✅ SUCCESS")
            print(f"     Requested: {width}x{height} @ {fps}fps")
            print(f"     Actual:    {actual_width}x{actual_height} @ {actual_fps}fps")
            print(f"     Measured:  {actual_fps_measured:.1f}fps (avg frame time: {avg_frame_time*1000:.1f}ms)")
        else:
            print(f"  ❌ FAILED ({success_count}/{total_tests} frames captured)")
    
    # Generate recommendations
    print("\n" + "=" * 50)
    print("📋 RECOMMENDATIONS")
    print("=" * 50)
    
    if working_configs:
        best_config = working_configs[0]  # First working config is highest preference
        
        print(f"\n🏆 RECOMMENDED SETTINGS for your system:")
        print(f"   camera_width = {best_config['actual'][0]}")
        print(f"   camera_height = {best_config['actual'][1]}")
        print(f"   camera_fps = {best_config['actual'][2]}")
        print(f"   camera_index = {camera_idx}")
        
        print(f"\n📝 Configuration description: {best_config['description']}")
        print(f"   Expected performance: {best_config['measured_fps']:.1f} fps")
        
        print(f"\n🔧 To use these settings, update your .env file:")
        print(f"   CAMERA_WIDTH={best_config['actual'][0]}")
        print(f"   CAMERA_HEIGHT={best_config['actual'][1]}")
        print(f"   CAMERA_FPS={best_config['actual'][2]}")
        print(f"   CAMERA_INDEX={camera_idx}")
        
        if len(working_configs) > 1:
            print(f"\n📊 Other working configurations:")
            for i, config in enumerate(working_configs[1:], 1):
                print(f"   {i+1}. {config['description']}: {config['actual'][0]}x{config['actual'][1]} @ {config['actual'][2]}fps")
    else:
        print("❌ No configurations worked! Your camera may have hardware issues.")
        print("   Try:")
        print("   1. Checking camera permissions in System Preferences")
        print("   2. Closing other applications using the camera")
        print("   3. Restarting your computer")
    
    print(f"\n💡 Camera auto-detection is enabled in Osmo Assistant, so it should")
    print(f"   automatically fall back to working settings even if you don't")
    print(f"   manually configure these values.")

if __name__ == "__main__":
    try:
        test_camera_configurations()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during camera testing: {e}")
        import traceback
        traceback.print_exc() 