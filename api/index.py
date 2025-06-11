from flask import Flask, request, jsonify, Response, render_template, send_file
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
import io

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MOBILE_MAX_RESOLUTION = (1080, 1920)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory storage for processing status and results
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
    """Apply phonk-style visual effects to frame"""
    frame = frame.astype(np.float32)
    
    # Increase contrast
    contrast = 1.3
    frame = ((frame - 128) * contrast + 128)
    
    # Boost saturation
    gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    gray = np.expand_dims(gray, axis=2)
    frame[...,:3] = gray + (frame[...,:3] - gray) * 1.25
    
    # Add purple/pink tint
    frame[..., 0] += 8   # Red
    frame[..., 2] += 12  # Blue
    
    # Simple vignette effect
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    y_indices, x_indices = np.ogrid[:h, :w]
    distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    normalized_distances = distances / max_distance
    vignette_strength = 0.4
    vignette_mask = 1 - (normalized_distances * vignette_strength)
    vignette_mask = np.expand_dims(vignette_mask, axis=2)
    
    frame = frame * vignette_mask
    frame = np.clip(frame, 0, 255)
    
    return frame.astype(np.uint8)

def process_video_in_memory(video_data, original_filename, job_id):
    """Process video entirely in memory using /tmp directory"""
    
    # Update progress
    processing_jobs[job_id]['progress'] = 10
    processing_jobs[job_id]['message'] = 'Loading video...'
    
    # Create temporary files in /tmp directory
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='/tmp') as temp_input:
        temp_input.write(video_data)
        temp_input_path = temp_input.name
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir='/tmp') as temp_output:
        temp_output_path = temp_output.name
    
    try:
        # Set environment variables for MoviePy
        os.environ['TMPDIR'] = '/tmp'
        os.environ['TEMP'] = '/tmp'
        os.environ['TMP'] = '/tmp'
        
        processing_jobs[job_id]['progress'] = 20
        processing_jobs[job_id]['message'] = 'Analyzing video...'
        
        # Load video
        clip = VideoFileClip(temp_input_path)
        
        processing_jobs[job_id]['progress'] = 30
        processing_jobs[job_id]['message'] = 'Resizing for mobile...'
        
        # Optimize for mobile
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
            
            clip = clip.resized((new_w, new_h))
        
        processing_jobs[job_id]['progress'] = 40
        processing_jobs[job_id]['message'] = 'Applying phonk effects...'
        
        def transform_frame(get_frame, t):
            """Transform each frame with phonk effects"""
            fade_duration = 3.0
            
            # Reverse playback
            reverse_t = clip.duration - t - 1/clip.fps
            reverse_t = max(0, min(reverse_t, clip.duration - 1/clip.fps))
            
            frame = get_frame(reverse_t)
            mirrored_frame = np.fliplr(frame)  # Mirror horizontally
            
            # Apply phonk effects
            phonk_frame = apply_phonk_effects(mirrored_frame)
            
            # Add flicker effect
            flicker_intensity = 0.97 + 0.03 * np.sin(t * 25)
            phonk_frame = (phonk_frame * flicker_intensity).astype(np.uint8)
            
            # Apply fade in
            if t < fade_duration:
                alpha = t / fade_duration
                return (phonk_frame * alpha).astype(np.uint8)
            else:
                return phonk_frame
        
        # Apply transformation
        final_clip = clip.transform(transform_frame)
        
        processing_jobs[job_id]['progress'] = 60
        processing_jobs[job_id]['message'] = 'Processing audio...'
        
        # Apply audio effects
        if clip.audio is not None:
            audio = clip.audio.with_volume_scaled(0.85)
            final_clip = final_clip.with_audio(audio)
        
        processing_jobs[job_id]['progress'] = 70
        processing_jobs[job_id]['message'] = 'Rendering final video...'
        
        # Write to temporary output file
        final_clip.write_videofile(
            temp_output_path,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            threads=2,
            bitrate="1500k",
            audio_bitrate="128k",
            logger=None,
            temp_audiofile=f'/tmp/temp_audio_{os.getpid()}.m4a'
        )
        
        processing_jobs[job_id]['progress'] = 90
        processing_jobs[job_id]['message'] = 'Finalizing...'
        
        # Read processed video into memory
        with open(temp_output_path, 'rb') as f:
            processed_video_data = f.read()
        
        # Clean up clips
        clip.close()
        final_clip.close()
        
        processing_jobs[job_id]['progress'] = 100
        
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
    """Process video in background thread"""
    try:
        processing_jobs[job_id]['status'] = JobStatus.PROCESSING
        processing_jobs[job_id]['message'] = 'Starting video processing...'
        processing_jobs[job_id]['progress'] = 0
        
        # Process the video
        processed_video_data = process_video_in_memory(video_data, original_filename, job_id)
        
        # Store the result
        processed_videos[job_id] = {
            'data': processed_video_data,
            'filename': f"phonk_{original_filename.rsplit('.', 1)[0]}.mp4",
            'created_at': time.time(),
            'size': len(processed_video_data)
        }
        
        # Update job status
        processing_jobs[job_id]['status'] = JobStatus.COMPLETED
        processing_jobs[job_id]['message'] = 'Video processing completed successfully!'
        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['download_url'] = f'/download/{job_id}'
        
        print(f"Job {job_id} completed successfully. Video size: {len(processed_video_data)} bytes")
        
    except Exception as e:
        processing_jobs[job_id]['status'] = JobStatus.ERROR
        processing_jobs[job_id]['message'] = f'Processing failed: {str(e)}'
        processing_jobs[job_id]['progress'] = 0
        print(f"Background processing error for job {job_id}: {str(e)}")

def cleanup_old_jobs():
    """Clean up old jobs and processed videos"""
    current_time = time.time()
    max_age = 3600  # 1 hour
    
    jobs_to_remove = []
    for job_id, job_data in processing_jobs.items():
        if current_time - job_data.get('created_at', 0) > max_age:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        processing_jobs.pop(job_id, None)
        processed_videos.pop(job_id, None)
        print(f"Cleaned up old job: {job_id}")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Read file data into memory
        video_data = file.read()
        
        if len(video_data) == 0:
            return jsonify({'error': 'Empty file uploaded'}), 400
        
        print(f"Received file: {file.filename}, size: {len(video_data)} bytes")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        processing_jobs[job_id] = {
            'status': JobStatus.PENDING,
            'message': 'Video uploaded successfully, processing will start shortly',
            'progress': 0,
            'created_at': time.time(),
            'original_filename': file.filename,
            'file_size': len(video_data)
        }
        
        # Start background processing
        thread = threading.Thread(
            target=background_process_video,
            args=(job_id, video_data, file.filename)
        )
        thread.daemon = True
        thread.start()
        
        print(f"Started processing job: {job_id}")
        
        # Return job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': JobStatus.PENDING,
            'message': 'Video uploaded successfully. Processing started.',
            'status_url': f'/status/{job_id}',
            'progress': 0
        }), 202
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<job_id>')
def check_status(job_id):
    """Check the status of a processing job"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job_data = processing_jobs[job_id].copy()
    
    # Add timing information
    elapsed = time.time() - job_data['created_at']
    job_data['elapsed_seconds'] = int(elapsed)
    
    # Estimate remaining time for processing jobs
    if job_data['status'] == JobStatus.PROCESSING:
        progress = job_data.get('progress', 0)
        if progress > 0:
            estimated_total = (elapsed / progress) * 100
            remaining = max(0, estimated_total - elapsed)
            job_data['estimated_remaining_seconds'] = int(remaining)
    
    return jsonify(job_data)

@app.route('/download/<job_id>')
def download_video(job_id):
    """Download the processed video"""
    print(f"Download request for job: {job_id}")
    
    if job_id not in processed_videos:
        if job_id in processing_jobs:
            status = processing_jobs[job_id]['status']
            if status == JobStatus.PROCESSING:
                return jsonify({'error': 'Video still processing', 'status': 'processing'}), 202
            elif status == JobStatus.ERROR:
                error_msg = processing_jobs[job_id].get('message', 'Video processing failed')
                return jsonify({'error': error_msg, 'status': 'error'}), 500
            else:
                return jsonify({'error': 'Video not ready', 'status': status}), 404
        else:
            return jsonify({'error': 'Job not found'}), 404
    
    video_info = processed_videos[job_id]
    print(f"Serving video: {video_info['filename']}, size: {video_info['size']} bytes")
    
    # Create a BytesIO object from the video data
    video_buffer = io.BytesIO(video_info['data'])
    video_buffer.seek(0)
    
    def cleanup_after_download():
        """Clean up after a short delay to ensure download completes"""
        time.sleep(5)  # Wait 5 seconds
        processed_videos.pop(job_id, None)
        processing_jobs.pop(job_id, None)
        print(f"Cleaned up job {job_id} after download")
    
    # Start cleanup in background
    cleanup_thread = threading.Thread(target=cleanup_after_download)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    return send_file(
        video_buffer,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=video_info['filename']
    )

@app.route('/jobs')
def list_jobs():
    """List all current jobs (for debugging)"""
    cleanup_old_jobs()
    return jsonify({
        'processing_jobs': len(processing_jobs),
        'processed_videos': len(processed_videos),
        'jobs': {job_id: {k:v for k,v in job_data.items() if k != 'data'} 
                for job_id, job_data in processing_jobs.items()}
    })

@app.route('/app-status')
def app_status():
    """Get application status"""
    cleanup_old_jobs()
    return jsonify({
        'status': 'running',
        'message': 'Async Phonk video processor is ready',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024),
        'active_jobs': len(processing_jobs),
        'completed_videos': len(processed_videos),
        'server_time': time.time()
    })

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024 * 1024)}MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    print("Starting Async Phonk Video Processor...")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB")
    print("Processing videos asynchronously - optimized for deployment")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
