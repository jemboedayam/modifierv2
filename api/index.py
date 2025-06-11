from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
import numpy as np
from moviepy import VideoFileClip
import tempfile
import os
import threading
import time
import uuid
import base64
import json

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MOBILE_MAX_RESOLUTION = (1080, 1920)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory storage for processing status and results
# In production, you'd want to use Redis or a database
processing_jobs = {}
processed_videos = {}

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_phonk_effects(frame):
    """Apply phonk-style visual effects to frame - optimized for speed"""
    frame = frame.astype(np.float32)
    
    # Faster contrast adjustment
    frame = frame * 1.3 - 128 * 0.3
    
    # Simplified saturation boost
    gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    gray = np.expand_dims(gray, axis=2)
    frame[...,:3] = gray + (frame[...,:3] - gray) * 1.2
    
    # Add purple/pink tint
    frame[..., 0] += 6   # Red
    frame[..., 2] += 10  # Blue
    
    # Simplified vignette effect
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    max_distance = min(center_x, center_y)
    
    # Create circular mask more efficiently
    y, x = np.ogrid[:h, :w]
    mask = ((x - center_x)**2 + (y - center_y)**2) <= (max_distance * 1.2)**2
    
    # Apply vignette only to edges
    frame[~mask] *= 0.7
    
    frame = np.clip(frame, 0, 255)
    return frame.astype(np.uint8)

def process_video_in_memory(video_data, original_filename):
    """Process video entirely in memory using /tmp directory for Vercel"""
    
    # Create temporary files in /tmp directory (required for Vercel)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='/tmp') as temp_input:
        temp_input.write(video_data)
        temp_input_path = temp_input.name
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='/tmp') as temp_output:
        temp_output_path = temp_output.name
    
    try:
        # Set TMPDIR environment variable to /tmp for MoviePy
        os.environ['TMPDIR'] = '/tmp'
        os.environ['TEMP'] = '/tmp'
        os.environ['TMP'] = '/tmp'
        
        # Load video
        clip = VideoFileClip(temp_input_path)
        
        # Optimize for mobile (faster resizing)
        w, h = clip.size
        max_w, max_h = MOBILE_MAX_RESOLUTION
        
        if w > max_w or h > max_h:
            ratio_w = max_w / w
            ratio_h = max_h / h
            ratio = min(ratio_w, ratio_h)
            
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            # Ensure even dimensions
            new_w = new_w if new_w % 2 == 0 else new_w - 1
            new_h = new_h if new_h % 2 == 0 else new_h - 1
            
            # Use faster resizing method
            clip = clip.resized((new_w, new_h)).with_fps(24)  # Lower FPS for faster processing
        else:
            # Still reduce FPS even if no resizing needed
            clip = clip.with_fps(24)
        
        def transform_frame(get_frame, t):
            """Transform each frame with phonk effects - optimized"""
            fade_duration = 2.0  # Shorter fade for faster processing
            
            # Reverse playback
            reverse_t = clip.duration - t - 1/clip.fps
            reverse_t = max(0, min(reverse_t, clip.duration - 1/clip.fps))
            
            frame = get_frame(reverse_t)
            mirrored_frame = np.fliplr(frame)  # Mirror horizontally
            
            # Apply phonk effects
            phonk_frame = apply_phonk_effects(mirrored_frame)
            
            # Simplified flicker effect
            flicker_intensity = 0.95 + 0.05 * np.sin(t * 20)
            phonk_frame = (phonk_frame * flicker_intensity).astype(np.uint8)
            
            # Apply fade in
            if t < fade_duration:
                alpha = t / fade_duration
                return (phonk_frame * alpha).astype(np.uint8)
            else:
                return phonk_frame
        
        # Apply transformation
        final_clip = clip.transform(transform_frame)
        
        # Apply audio effects (simplified for speed)
        if clip.audio is not None:
            audio = clip.audio.with_volume_scaled(0.8)  # Simple volume adjustment
            final_clip = final_clip.with_audio(audio)
        
        # Limit video duration for faster processing (optional)
        max_duration = 30  # seconds
        if final_clip.duration > max_duration:
            final_clip = final_clip.subclipped(0, max_duration)
        
        # Write to temporary output file in /tmp with optimized settings for speed
        final_clip.write_videofile(
            temp_output_path,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",  # Fastest encoding preset
            threads=4,  # More threads for faster processing
            bitrate="1000k",  # Lower bitrate for faster processing
            audio_bitrate="96k",  # Lower audio bitrate
            logger=None,
            verbose=False,  # Disable verbose output
            temp_audiofile=f'/tmp/temp_audio_{os.getpid()}.m4a',
            ffmpeg_params=['-crf', '28', '-movflags', '+faststart']  # Fast encoding params
        )
        
        # Read processed video into memory
        with open(temp_output_path, 'rb') as f:
            processed_video_data = f.read()
        
        # Clean up clips
        clip.close()
        final_clip.close()
        
        return processed_video_data
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            temp_audio_path = f'/tmp/temp_audio_{os.getpid()}.m4a'
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception as cleanup_error:
            print(f"Cleanup warning: {cleanup_error}")

def background_process_video(job_id, video_data, original_filename):
    """Process video in background thread - optimized for speed"""
    try:
        processing_jobs[job_id]['status'] = JobStatus.PROCESSING
        processing_jobs[job_id]['message'] = 'Processing video with phonk effects...'
        
        # Process the video
        processed_video_data = process_video_in_memory(video_data, original_filename)
        
        # Store the result (no need for base64 anymore)
        processed_videos[job_id] = {
            'data': processed_video_data,
            'filename': f"phonk_{original_filename.rsplit('.', 1)[0]}.mp4",
            'created_at': time.time(),
            'size_bytes': len(processed_video_data)
        }
        
        # Update job status
        processing_jobs[job_id]['status'] = JobStatus.COMPLETED
        processing_jobs[job_id]['message'] = 'Video processing completed - ready for download'
        processing_jobs[job_id]['download_url'] = f'/download-auto/{job_id}'
        processing_jobs[job_id]['filename'] = f"phonk_{original_filename.rsplit('.', 1)[0]}.mp4"
        processing_jobs[job_id]['size_bytes'] = len(processed_video_data)
        processing_jobs[job_id]['processing_time'] = time.time() - processing_jobs[job_id]['created_at']
        
        print(f"✅ Job {job_id} completed in {processing_jobs[job_id]['processing_time']:.1f}s")
        
    except Exception as e:
        processing_jobs[job_id]['status'] = JobStatus.ERROR
        processing_jobs[job_id]['message'] = f'Processing failed: {str(e)}'
        print(f"❌ Background processing error for job {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()

def cleanup_old_jobs():
    """Clean up old jobs and processed videos (runs periodically)"""
    current_time = time.time()
    max_age = 3600  # 1 hour
    
    # Clean up old processing jobs
    jobs_to_remove = []
    for job_id, job_data in processing_jobs.items():
        if current_time - job_data.get('created_at', 0) > max_age:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        processing_jobs.pop(job_id, None)
        processed_videos.pop(job_id, None)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read file data into memory
        video_data = file.read()
        
        if len(video_data) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        processing_jobs[job_id] = {
            'status': JobStatus.PENDING,
            'message': 'Video uploaded, processing will start shortly',
            'created_at': time.time(),
            'original_filename': file.filename
        }
        
        # Start background processing
        thread = threading.Thread(
            target=background_process_video,
            args=(job_id, video_data, file.filename)
        )
        thread.daemon = True
        thread.start()
        
        # Return job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': JobStatus.PENDING,
            'message': 'Video uploaded successfully. Processing started.',
            'status_url': f'/status/{job_id}'
        }), 202
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<job_id>')
def check_status(job_id):
    """Check the status of a processing job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job_data = processing_jobs[job_id].copy()
    
    # Add estimated time remaining for processing jobs
    if job_data['status'] == JobStatus.PROCESSING:
        elapsed = time.time() - job_data['created_at']
        job_data['elapsed_seconds'] = int(elapsed)
        job_data['estimated_total_seconds'] = 120  # Rough estimate
    
    # If completed, include download info for auto-download
    if job_data['status'] == JobStatus.COMPLETED and job_id in processed_videos:
        job_data['ready_for_download'] = True
        job_data['download_ready'] = True
    
    return jsonify(job_data)

@app.route('/download/<job_id>')
def download_video(job_id):
    """Download the processed video"""
    if job_id not in processed_videos:
        if job_id in processing_jobs:
            status = processing_jobs[job_id]['status']
            if status == JobStatus.PROCESSING:
                return jsonify({'error': 'Video still processing'}), 202
            elif status == JobStatus.ERROR:
                return jsonify({'error': 'Video processing failed'}), 500
            else:
                return jsonify({'error': 'Video not ready'}), 404
        else:
            return jsonify({'error': 'Job not found'}), 404
    
    video_info = processed_videos[job_id]
    
    def generate():
        yield video_info['data']
        # Clean up after download
        processed_videos.pop(job_id, None)
        processing_jobs.pop(job_id, None)
    
    response = Response(
        generate(),
        mimetype='video/mp4',
        headers={
            'Content-Disposition': f'attachment; filename="{video_info["filename"]}"',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )
    
    return response

@app.route('/download-auto/<job_id>')
def download_video_auto(job_id):
    """Auto-download endpoint that returns video data for immediate download"""
    if job_id not in processed_videos:
        if job_id in processing_jobs:
            status = processing_jobs[job_id]['status']
            if status == JobStatus.PROCESSING:
                return jsonify({'error': 'Video still processing'}), 202
            elif status == JobStatus.ERROR:
                return jsonify({'error': 'Video processing failed', 'message': processing_jobs[job_id]['message']}), 500
            else:
                return jsonify({'error': 'Video not ready'}), 404
        else:
            return jsonify({'error': 'Job not found'}), 404
    
    video_info = processed_videos[job_id]
    
    # Return video as direct download response instead of base64
    def generate():
        yield video_info['data']
        # Clean up after providing download data
        processed_videos.pop(job_id, None)
        processing_jobs.pop(job_id, None)
    
    response = Response(
        generate(),
        mimetype='video/mp4',
        headers={
            'Content-Disposition': f'attachment; filename="{video_info["filename"]}"',
            'Content-Length': str(video_info['size_bytes']),
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Access-Control-Expose-Headers': 'Content-Disposition'
        }
    )
    
    return response

@app.route('/jobs')
def list_jobs():
    """List all current jobs (for debugging)"""
    cleanup_old_jobs()
    return jsonify({
        'processing_jobs': len(processing_jobs),
        'processed_videos': len(processed_videos),
        'jobs': {job_id: {k:v for k,v in job_data.items() if k not in ['data', 'data_base64']} 
                for job_id, job_data in processing_jobs.items()}
    })

@app.route('/app-status')
def app_status():
    cleanup_old_jobs()
    return jsonify({
        'status': 'running',
        'message': 'Async Phonk video processor is ready',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024),
        'active_jobs': len(processing_jobs),
        'completed_videos': len(processed_videos)
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Async Phonk Video Processor...")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB")
    print("Processing videos asynchronously - optimized for Vercel with auto-download")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
