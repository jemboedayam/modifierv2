from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
import numpy as np
from moviepy import VideoFileClip
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MOBILE_MAX_RESOLUTION = (1080, 1920)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

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
        
        # Apply audio effects
        if clip.audio is not None:
            audio = clip.audio.with_volume_scaled(0.85)
            final_clip = final_clip.with_audio(audio)
        
        # Write to temporary output file in /tmp
        final_clip.write_videofile(
            temp_output_path,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=2,
            bitrate="2000k",
            audio_bitrate="128k",
            logger=None,
            temp_audiofile=f'/tmp/temp_audio_{os.getpid()}.m4a'  # Explicitly set temp audio path
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
            # Clean up any additional temp files MoviePy might have created
            temp_audio_path = f'/tmp/temp_audio_{os.getpid()}.m4a'
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception as cleanup_error:
            print(f"Cleanup warning: {cleanup_error}")

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
        
        # Process video in memory
        try:
            processed_video_data = process_video_in_memory(video_data, file.filename)
            
            # Create response with processed video
            def generate():
                yield processed_video_data
                # Memory is automatically freed when generator completes
            
            # Generate processed filename
            name_without_ext = file.filename.rsplit('.', 1)[0]
            processed_filename = f"phonk_{name_without_ext}.mp4"
            
            response = Response(
                generate(),
                mimetype='video/mp4',
                headers={
                    'Content-Disposition': f'attachment; filename="{processed_filename}"',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
            
            return response
            
        except Exception as e:
            return jsonify({'error': f'Video processing failed: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'message': 'In-memory Phonk video processor is ready (Vercel optimized)',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024)
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting In-Memory Phonk Video Processor...")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB")
    print("Processing videos entirely in memory - optimized for Vercel")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
