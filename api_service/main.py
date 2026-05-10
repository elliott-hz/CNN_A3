"""
Dog Emotion Recognition API Service
FastAPI backend for dog face detection and emotion classification
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator
import torch
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
import base64
import cv2
import tempfile
import os
import uuid
import json
import asyncio

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.pipeline_inference import PipelineInference


# Pydantic models for API response
class DetectionResult(BaseModel):
    dog_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    detection_confidence: float
    emotion: str
    emotion_confidence: float
    emotion_probabilities: Dict[str, float]


class InferenceResponse(BaseModel):
    success: bool
    results: List[DetectionResult]
    message: str = ""


class Base64ImageRequest(BaseModel):
    image_base64: str


class FrameDetection(BaseModel):
    timestamp: float
    detections: List[DetectionResult]


class VideoAnalysisResponse(BaseModel):
    success: bool
    video_duration: float
    frame_interval: float
    total_frames: int
    frames: List[FrameDetection]
    message: str = ""


class ProgressUpdate(BaseModel):
    """Progress update for streaming"""
    progress: float  # 0-100
    current_frame: int
    total_frames: int
    status: str  # "processing", "complete", "error"
    message: str = ""


# Initialize FastAPI app
app = FastAPI(
    title="Dog Emotion Recognition API",
    description="API for dog face detection and emotion classification",
    version="1.0.0"
)

# CORS middleware - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global pipeline instance (loaded once at startup)
pipeline = None


@app.on_event("startup")
async def load_models():
    """Load models when the application starts"""
    global pipeline
    
    print("=" * 80)
    print("Loading Dog Emotion Recognition Models...")
    print("=" * 80)
    
    try:
        # Model paths
        detection_model_path = Path(__file__).parent.parent / "best_models" / "detection_YOLOv8_baseline.pt"
        classification_model_path = Path(__file__).parent.parent / "best_models" / "emotion_ResNet50_baseline.pth"
        
        # Verify model files exist
        if not detection_model_path.exists():
            raise FileNotFoundError(f"Detection model not found: {detection_model_path}")
        if not classification_model_path.exists():
            raise FileNotFoundError(f"Classification model not found: {classification_model_path}")
        
        # Initialize pipeline
        pipeline = PipelineInference(
            detection_model_path=str(detection_model_path),
            classification_model_path=str(classification_model_path)
        )
        
        # Check device
        if torch.cuda.is_available():
            device = "GPU (CUDA)"
        elif torch.backends.mps.is_available():
            device = "GPU (MPS - Apple Silicon)"
        else:
            device = "CPU"
        print(f"\n✅ Models loaded successfully!")
        print(f"🖥️  Running on: {device}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Dog Emotion Recognition API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if torch.cuda.is_available():
        device = "GPU (CUDA)"
    elif torch.backends.mps.is_available():
        device = "GPU (MPS - Apple Silicon)"
    else:
        device = "CPU"
    
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "models_loaded": pipeline is not None,
        "device": device
    }


@app.post("/api/detect", response_model=InferenceResponse)
async def detect_emotion(file: UploadFile = File(...)):
    """
    Detect dog faces and classify emotions in uploaded image.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        List of detections with bounding boxes and emotion labels
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        contents = await file.read()
        
        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Open image with PIL
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to temporary file for pipeline (YOLOv8 expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, format='JPEG')
            temp_path = tmp.name
        
        try:
            # Run inference
            results = pipeline.predict(temp_path, conf=0.5, iou=0.45)
            
            # Format response
            detection_results = []
            for result in results:
                detection_results.append(DetectionResult(
                    dog_id=result['dog_id'],
                    bbox=result['bbox'],
                    detection_confidence=result['detection_confidence'],
                    emotion=result['emotion'],
                    emotion_confidence=result['emotion_confidence'],
                    emotion_probabilities=result['emotion_probabilities']
                ))
            
            message = f"Detected {len(detection_results)} dog(s)" if detection_results else "No dogs detected"
            
            return InferenceResponse(
                success=True,
                results=detection_results,
                message=message
            )
        
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/detect-base64", response_model=InferenceResponse)
async def detect_emotion_base64(request: Base64ImageRequest):
    """
    Detect dog faces and classify emotions from base64 encoded image.
    Optimized for video frame processing.
    
    Args:
        request: Base64ImageRequest containing base64 encoded image string
        
    Returns:
        List of detections with bounding boxes and emotion labels
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to temporary file for pipeline (YOLOv8 expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, format='JPEG')
            temp_path = tmp.name
        
        try:
            # Run inference
            results = pipeline.predict(temp_path, conf=0.5, iou=0.45)
            
            # Format response
            detection_results = []
            for result in results:
                detection_results.append(DetectionResult(
                    dog_id=result['dog_id'],
                    bbox=result['bbox'],
                    detection_confidence=result['detection_confidence'],
                    emotion=result['emotion'],
                    emotion_confidence=result['emotion_confidence'],
                    emotion_probabilities=result['emotion_probabilities']
                ))
            
            message = f"Detected {len(detection_results)} dog(s)" if detection_results else "No dogs detected"
            
            return InferenceResponse(
                success=True,
                results=detection_results,
                message=message
            )
        
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze video file for dog faces and emotions.
    Processes frames at 5fps (every 200ms) for videos up to 20 seconds.
    
    Args:
        file: Uploaded video file (MP4, WebM, AVI, max 20 seconds)
        
    Returns:
        Analysis results with timestamps and detections (100 frames max)
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read video data
        contents = await file.read()
        
        # Check file size (max 50MB)
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Video too large (max 50MB)")
        
        # Save to temporary file for processing
        temp_video_path = f"{tempfile.gettempdir()}/{uuid.uuid4()}.mp4"
        with open(temp_video_path, 'wb') as f:
            f.write(contents)
        
        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Failed to open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            # Validate video duration (max 20 seconds)
            if video_duration > 20.0:
                cap.release()
                raise HTTPException(
                    status_code=400, 
                    detail=f"Video too long ({video_duration:.1f}s). Maximum duration is 20 seconds."
                )
            
            # Set sampling rate: 5 frames per second (every 200ms)
            target_fps = 5.0
            frame_interval = 1.0 / target_fps  # 0.2 seconds
            
            # Calculate which frames to sample
            # For a 20fps video, we sample every 4th frame to get 5fps
            sample_interval = int(fps / target_fps) if fps > 0 else 1
            sample_interval = max(1, sample_interval)  # At least every frame
            
            print(f"Video properties: {fps}fps, {total_frames} frames, {video_duration:.2f}s duration")
            print(f"Sampling at {target_fps}fps (every {sample_interval} frames)")
            
            # Initialize frame analysis list
            frames = []
            frame_count = 0
            sampled_frame_count = 0
            
            # Process frames at target sampling rate
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process frames at our target sampling rate
                if frame_count % sample_interval == 0:
                    timestamp = sampled_frame_count * frame_interval
                    
                    # Stop if we exceed 20 seconds
                    if timestamp > 20.0:
                        break
                    
                    try:
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Save frame to temporary file for pipeline (YOLOv8 expects file path)
                        temp_frame_path = f"{tempfile.gettempdir()}/{uuid.uuid4()}.jpg"
                        cv2.imwrite(temp_frame_path, frame_rgb)
                        
                        # Run inference
                        results = pipeline.predict(temp_frame_path, conf=0.5, iou=0.45)
                        
                        # Format response
                        detection_results = []
                        for result in results:
                            detection_results.append(DetectionResult(
                                dog_id=result['dog_id'],
                                bbox=result['bbox'],
                                detection_confidence=result['detection_confidence'],
                                emotion=result['emotion'],
                                emotion_confidence=result['emotion_confidence'],
                                emotion_probabilities=result['emotion_probabilities']
                            ))
                        
                        # Append frame analysis to list
                        frames.append(FrameDetection(
                            timestamp=timestamp,
                            detections=detection_results
                        ))
                        
                        sampled_frame_count += 1
                        
                        # Log progress every 10 frames
                        if sampled_frame_count % 10 == 0:
                            print(f"Processed {sampled_frame_count} frames...")
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                    finally:
                        # Clean up temporary frame file
                        if 'temp_frame_path' in locals() and os.path.exists(temp_frame_path):
                            os.unlink(temp_frame_path)
                
                frame_count += 1
            
            # Release video capture
            cap.release()
            
            actual_duration = sampled_frame_count * frame_interval
            
            message = f"Analyzed {sampled_frame_count} frame(s) at {target_fps}fps over {actual_duration:.1f} seconds"
            print(f"Analysis complete: {message}")
            
            return VideoAnalysisResponse(
                success=True,
                video_duration=actual_duration,
                frame_interval=frame_interval,
                total_frames=sampled_frame_count,
                frames=frames,
                message=message
            )
        
        finally:
            # Clean up temporary video file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@app.post("/api/analyze-video-batch", response_model=VideoAnalysisResponse)
async def analyze_video_batch(file: UploadFile = File(...), batch_size: int = 10):
    """
    Optimized video analysis using batch processing.
    Extracts all frames first, then processes them in batches for maximum speed.
    
    Args:
        file: Uploaded video file (MP4, WebM, AVI, max 20 seconds)
        batch_size: Number of frames to process simultaneously (default: 10)
        
    Returns:
        Analysis results with timestamps and detections
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read video data
        contents = await file.read()
        
        # Check file size (max 50MB)
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Video too large (max 50MB)")
        
        # Save to temporary file for processing
        temp_video_path = f"{tempfile.gettempdir()}/{uuid.uuid4()}.mp4"
        with open(temp_video_path, 'wb') as f:
            f.write(contents)
        
        try:
            print(f"\n{'='*80}")
            print("BATCH PROCESSING MODE")
            print(f"{'='*80}")
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Failed to open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            
            # Validate video duration (max 20 seconds)
            if video_duration > 20.0:
                cap.release()
                raise HTTPException(
                    status_code=400, 
                    detail=f"Video too long ({video_duration:.1f}s). Maximum duration is 20 seconds."
                )
            
            # Set sampling rate: 5 frames per second (every 200ms)
            target_fps = 5.0
            frame_interval = 1.0 / target_fps  # 0.2 seconds
            
            # Calculate which frames to sample
            sample_interval = int(fps / target_fps) if fps > 0 else 1
            sample_interval = max(1, sample_interval)
            
            print(f"Video properties: {fps}fps, {total_frames} frames, {video_duration:.2f}s duration")
            print(f"Sampling at {target_fps}fps (every {sample_interval} frames)")
            print(f"Batch size: {batch_size} frames")
            
            # STEP 1: Extract all frames into memory (fast sequential read)
            print("\n[Step 1/3] Extracting frames from video...")
            frames_data = []  # List of (timestamp, frame_array)
            frame_count = 0
            sampled_frame_count = 0
            
            import time
            extract_start = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only sample frames at target rate
                if frame_count % sample_interval == 0:
                    timestamp = sampled_frame_count * frame_interval
                    
                    # Stop if we exceed 20 seconds
                    if timestamp > 20.0:
                        break
                    
                    # Store frame in memory (BGR format)
                    frames_data.append((timestamp, frame.copy()))
                    sampled_frame_count += 1
                
                frame_count += 1
            
            cap.release()
            extract_time = time.time() - extract_start
            print(f"✓ Extracted {len(frames_data)} frames in {extract_time:.2f}s")
            
            if not frames_data:
                return VideoAnalysisResponse(
                    success=True,
                    video_duration=0.0,
                    frame_interval=frame_interval,
                    total_frames=0,
                    frames=[],
                    message="No frames extracted from video"
                )
            
            # STEP 2: Process frames in batches using true batch inference
            print(f"\n[Step 2/3] Processing {len(frames_data)} frames in batches of {batch_size}...")
            inference_start = time.time()
            
            frames_results = []
            total_batches = (len(frames_data) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(frames_data), batch_size):
                batch_data = frames_data[batch_idx:batch_idx + batch_size]
                current_batch_num = (batch_idx // batch_size) + 1
                
                # Extract just the frames (numpy arrays) and timestamps
                batch_frames = [frame for (_, frame) in batch_data]
                batch_timestamps = [ts for (ts, _) in batch_data]
                
                print(f"  Processing batch {current_batch_num}/{total_batches} ({len(batch_frames)} frames)...")
                
                try:
                    # TRUE BATCH INFERENCE: Process all frames simultaneously!
                    batch_results = pipeline.predict_batch(batch_frames, conf=0.5, iou=0.45)
                    
                    # Format results for each frame
                    for frame_idx, (timestamp, detections) in enumerate(zip(batch_timestamps, batch_results)):
                        detection_results = []
                        for result in detections:
                            detection_results.append(DetectionResult(
                                dog_id=result['dog_id'],
                                bbox=result['bbox'],
                                detection_confidence=result['detection_confidence'],
                                emotion=result['emotion'],
                                emotion_confidence=result['emotion_confidence'],
                                emotion_probabilities=result['emotion_probabilities']
                            ))
                        
                        frames_results.append(FrameDetection(
                            timestamp=timestamp,
                            detections=detection_results
                        ))
                
                except Exception as e:
                    print(f"  ⚠ Error processing batch {current_batch_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Add empty detections for failed batch
                    for timestamp, _ in batch_data:
                        frames_results.append(FrameDetection(
                            timestamp=timestamp,
                            detections=[]
                        ))
                
                # Allow async event loop to process other tasks
                await asyncio.sleep(0)
            
            inference_time = time.time() - inference_start
            print(f"✓ Batch inference completed in {inference_time:.2f}s")
            print(f"  Average: {inference_time/len(frames_data)*1000:.0f}ms per frame")
            print(f"  Speedup: ~{len(frames_data)/inference_time:.1f} frames/sec")
            
            # STEP 3: Return results
            actual_duration = len(frames_results) * frame_interval
            
            total_time = extract_time + inference_time
            message = f"Analyzed {len(frames_results)} frame(s) at {target_fps}fps over {actual_duration:.1f} seconds (total: {total_time:.2f}s)"
            print(f"\n[Step 3/3] {message}")
            print(f"{'='*80}\n")
            
            return VideoAnalysisResponse(
                success=True,
                video_duration=actual_duration,
                frame_interval=frame_interval,
                total_frames=len(frames_results),
                frames=frames_results,
                message=message
            )
        
        finally:
            # Clean up temporary video file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@app.post("/api/analyze-video-stream")
async def analyze_video_stream(file: UploadFile = File(...)):
    """
    Analyze video file with real-time progress updates via Server-Sent Events (SSE).
    Optimized for speed - processes frames directly in memory without file I/O.
    
    Args:
        file: Uploaded video file (MP4, WebM, AVI, max 20 seconds)
        
    Returns:
        SSE stream with progress updates, followed by final results
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    async def generate_progress():
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('video/'):
                yield f"data: {json.dumps({'error': 'File must be a video'})}\n\n"
                return
            
            # Read video data
            contents = await file.read()
            
            # Check file size (max 50MB)
            if len(contents) > 50 * 1024 * 1024:
                yield f"data: {json.dumps({'error': 'Video too large (max 50MB)'})}\n\n"
                return
            
            # Save to temporary file for processing
            temp_video_path = f"{tempfile.gettempdir()}/{uuid.uuid4()}.mp4"
            with open(temp_video_path, 'wb') as f:
                f.write(contents)
            
            try:
                # Open video with OpenCV
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    yield f"data: {json.dumps({'error': 'Failed to open video file'})}\n\n"
                    return
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_frames / fps
                
                # Validate video duration (max 20 seconds)
                if video_duration > 20.0:
                    cap.release()
                    yield f"data: {json.dumps({'error': f'Video too long ({video_duration:.1f}s). Maximum duration is 20 seconds.'})}\n\n"
                    return
                
                # Set sampling rate: 5 frames per second (every 200ms)
                target_fps = 5.0
                frame_interval = 1.0 / target_fps  # 0.2 seconds
                
                # Calculate which frames to sample
                sample_interval = int(fps / target_fps) if fps > 0 else 1
                sample_interval = max(1, sample_interval)
                
                print(f"Video properties: {fps}fps, {total_frames} frames, {video_duration:.2f}s duration")
                print(f"Sampling at {target_fps}fps (every {sample_interval} frames)")
                
                # Calculate total frames to process
                total_sampled_frames = min(int(video_duration * target_fps), 100)
                
                # Initialize frame analysis list
                frames = []
                frame_count = 0
                sampled_frame_count = 0
                
                # Send initial progress
                yield f"data: {json.dumps({'progress': 0, 'current_frame': 0, 'total_frames': total_sampled_frames, 'status': 'processing', 'message': 'Starting analysis...'})}\n\n"
                
                # Process frames at target sampling rate
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Only process frames at our target sampling rate
                    if frame_count % sample_interval == 0:
                        timestamp = sampled_frame_count * frame_interval
                        
                        # Stop if we exceed 20 seconds
                        if timestamp > 20.0:
                            break
                        
                        try:
                            # OPTIMIZATION: Run inference directly on numpy array (no file I/O!)
                            results = pipeline.predict(frame, conf=0.5, iou=0.45)
                            
                            # Format response
                            detection_results = []
                            for result in results:
                                detection_results.append({
                                    'dog_id': result['dog_id'],
                                    'bbox': result['bbox'],
                                    'detection_confidence': result['detection_confidence'],
                                    'emotion': result['emotion'],
                                    'emotion_confidence': result['emotion_confidence'],
                                    'emotion_probabilities': result['emotion_probabilities']
                                })
                            
                            # Append frame analysis to list
                            frames.append({
                                'timestamp': timestamp,
                                'detections': detection_results
                            })
                            
                            sampled_frame_count += 1
                            
                            # Calculate and send progress update
                            progress = min(100.0, (sampled_frame_count / total_sampled_frames) * 100)
                            progress_data = {
                                'progress': round(progress, 1),
                                'current_frame': sampled_frame_count,
                                'total_frames': total_sampled_frames,
                                'status': 'processing',
                                'message': f'Processing frame {sampled_frame_count}/{total_sampled_frames}'
                            }
                            yield f"data: {json.dumps(progress_data)}\n\n"
                            
                            # Allow other async tasks to run
                            await asyncio.sleep(0)
                            
                        except Exception as e:
                            print(f"Error processing frame {frame_count}: {e}")
                    
                    frame_count += 1
                
                # Release video capture
                cap.release()
                
                actual_duration = sampled_frame_count * frame_interval
                
                message = f"Analyzed {sampled_frame_count} frame(s) at {target_fps}fps over {actual_duration:.1f} seconds"
                print(f"Analysis complete: {message}")
                
                # Send completion message
                completion_data = {
                    'progress': 100.0,
                    'current_frame': sampled_frame_count,
                    'total_frames': sampled_frame_count,
                    'status': 'complete',
                    'message': message,
                    'success': True,
                    'video_duration': actual_duration,
                    'frame_interval': frame_interval,
                    'frames': frames
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
            
            finally:
                # Clean up temporary video file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
        
        except Exception as e:
            print(f"Error during video analysis: {e}")
            import traceback
            traceback.print_exc()
            error_data = {
                'progress': 0,
                'current_frame': 0,
                'total_frames': 0,
                'status': 'error',
                'message': f'Video analysis failed: {str(e)}'
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")


@app.websocket("/ws/live-stream")
async def websocket_live_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live streaming inference.
    Receives frames from client, runs detection + classification, sends back results.
    
    Client should send base64-encoded frames and receive JSON responses with detections.
    
    Note: Requires GPU (CUDA or MPS) for acceptable performance. CPU will be very slow.
    """
    global pipeline
    
    # Check if pipeline is initialized
    if pipeline is None:
        await websocket.close(code=1011, reason="Models not loaded yet")
        return
    
    # Check origin for security (allow localhost development)
    origin = websocket.headers.get("origin", "")
    allowed_origins = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"]
    
    if origin and origin not in allowed_origins:
        print(f"⚠️  WebSocket connection rejected from origin: {origin}")
        await websocket.close(code=4003, reason="Origin not allowed")
        return
    
    # Check device capability
    if torch.cuda.is_available():
        device_info = "GPU (CUDA)"
    elif torch.backends.mps.is_available():
        device_info = "GPU (MPS - Apple Silicon)"
    else:
        device_info = "CPU"
        print("️  WARNING: Running on CPU - live streaming will be very slow!")
        print("💡 Recommendation: Use GPU (CUDA or MPS) for real-time performance")
    
    # Accept the WebSocket connection
    await websocket.accept()
    print(f"✅ Live stream connection established from {origin} - Using: {device_info}")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            try:
                # Parse incoming message
                message = json.loads(data)
                
                if 'frame' not in message:
                    await websocket.send_json({
                        'error': 'No frame data received',
                        'success': False
                    })
                    continue
                
                # Decode base64 image
                frame_base64 = message['frame']
                frame_bytes = base64.b64decode(frame_base64)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame_bgr is None:
                    await websocket.send_json({
                        'error': 'Failed to decode frame',
                        'success': False
                    })
                    continue
                
                # Convert to RGB for pipeline
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                try:
                    # OPTIMIZATION: Process directly from numpy array (skip file I/O)
                    # Pipeline supports both file paths and numpy arrays
                    results = pipeline.predict(frame_rgb, conf=0.5, iou=0.45)
                    
                    # Format response
                    response_data = {
                        'success': True,
                        'detections': [],
                        'device': device_info,
                        'timestamp': message.get('timestamp', 0)
                    }
                    
                    for result in results:
                        response_data['detections'].append({
                            'dog_id': result['dog_id'],
                            'bbox': result['bbox'],
                            'detection_confidence': result['detection_confidence'],
                            'emotion': result['emotion'],
                            'emotion_confidence': result['emotion_confidence'],
                            'emotion_probabilities': result['emotion_probabilities']
                        })
                    
                    # Send results back to client
                    await websocket.send_json(response_data)
                    
                except Exception as e:
                    print(f"Error during inference: {e}")
                    await websocket.send_json({
                        'error': str(e),
                        'success': False
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    'error': 'Invalid JSON format',
                    'success': False
                })
            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_json({
                    'error': str(e),
                    'success': False
                })
    
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        print("Live stream connection closed")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
