from flask import Flask, request, jsonify, send_file,render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from moviepy import VideoFileClip
import numpy as np
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration optimized for mobile
UPLOAD_FOLDER = 'single'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', '3gp', 'mp4v'}  # Added mobile formats
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # Reduced to 200MB for mobile compatibility
MOBILE_MAX_RESOLUTION = (1080, 1920)  # Mobile-friendly max resolution

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def optimize_for_mobile(clip):
    """Optimize video settings for mobile devices"""
    # Get current dimensions
    w, h = clip.size
    
    # Resize if too large for mobile
    max_w, max_h = MOBILE_MAX_RESOLUTION
    if w > max_w or h > max_h:
        # Calculate aspect ratio preserving resize
        ratio_w = max_w / w
        ratio_h = max_h / h
        ratio = min(ratio_w, ratio_h)
        
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Ensure dimensions are even (required for some codecs)
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1
        
        clip = clip.resized((new_w, new_h))
        print(f"Resized video from {w}x{h} to {new_w}x{new_h} for mobile compatibility")
    
    return clip

def modify_video(file_name, original_filename):
    """Apply phonk-style effects to video and return the processed file path - Mobile optimized"""
    import tempfile
    
    FILENAME = os.path.join("single", file_name)
    
    # Create a temporary file for the processed video
    temp_dir = tempfile.gettempdir()
    processed_filename = f"phonk_{original_filename}"
    to_write = os.path.join(temp_dir, processed_filename)

    clip = VideoFileClip(FILENAME)
    
    # Optimize for mobile before processing
    clip = optimize_for_mobile(clip)

    def apply_phonk_effects(frame):
        """Apply phonk-style visual effects to frame - Optimized for mobile processing"""
        frame = frame.astype(np.float32)
        
        # Slightly reduced effects for better mobile performance
        # Increase contrast (phonk style - high contrast)
        contrast = 1.3  # Reduced from 1.4 for better mobile performance
        frame = ((frame - 128) * contrast + 128)
        
        # Adjust saturation
        saturation_boost = 1.25  # Reduced from 1.3
        
        # Simple saturation boost by enhancing color differences from gray
        gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        gray = np.expand_dims(gray, axis=2)
        
        # Boost color deviation from grayscale
        frame[...,:3] = gray + (frame[...,:3] - gray) * saturation_boost
        
        # Add slight purple/pink tint (typical phonk aesthetic)
        frame[..., 0] += 8   # Reduced from 10
        frame[..., 2] += 12  # Reduced from 15
        
        # Simplified vignette effect for better mobile performance
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create simpler vignette mask
        y_indices, x_indices = np.ogrid[:h, :w]
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalized distance from center
        normalized_distances = distances / max_distance
        
        # Apply vignette (lighter effect for mobile)
        vignette_strength = 0.4  # Reduced from 0.6
        vignette_mask = 1 - (normalized_distances * vignette_strength)
        vignette_mask = np.expand_dims(vignette_mask, axis=2)
        
        frame = frame * vignette_mask
        
        # Clamp values
        frame = np.clip(frame, 0, 255)
        
        return frame.astype(np.uint8)

    def optimized_transform(get_frame, t):
        fade_duration = 3.0  # Reduced fade duration
        
        reverse_t = clip.duration - t - 1/clip.fps
        reverse_t = max(0, min(reverse_t, clip.duration - 1/clip.fps))
        
        frame = get_frame(reverse_t)
        mirrored_frame = np.fliplr(frame)
        
        # Apply phonk effects
        phonk_frame = apply_phonk_effects(mirrored_frame)
        
        # Add subtle flicker effect (reduced for mobile)
        flicker_intensity = 0.97 + 0.03 * np.sin(t * 25)  # Reduced intensity and frequency
        phonk_frame = (phonk_frame * flicker_intensity).astype(np.uint8)
        
        # Apply fade
        if t < fade_duration:
            alpha = t / fade_duration
            return (phonk_frame * alpha).astype(np.uint8)
        else:
            return phonk_frame

    final_clip = clip.transform(optimized_transform)

    # Apply audio effects for phonk style
    if clip.audio is not None:
        audio = clip.audio
        audio = audio.with_volume_scaled(0.85)  # Slightly adjusted
        final_clip = final_clip.with_audio(audio)

    final_clip = final_clip.with_duration(clip.duration)

    # Mobile-optimized export settings
    final_clip.write_videofile(
        to_write, 
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        preset="medium",  # Better compression for mobile
        threads=2,        # Reduced threads for mobile compatibility
        bitrate="2000k",  # Reasonable bitrate for mobile
        audio_bitrate="128k"  # Good audio quality for mobile
    )

    clip.close()
    final_clip.close()
    os.remove(FILENAME)  # Remove original uploaded file

    print(f"Mobile-optimized phonk video processed: {processed_filename}")
    print(f"Removed original file: {file_name}")
    
    return to_write, processed_filename

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename to avoid conflicts
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # Save the uploaded file temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            print(f"File uploaded successfully: {unique_filename}")
            
            # Apply phonk effects to the video
            try:
                processed_file_path, processed_filename = modify_video(unique_filename, file.filename)
                
                # Send the processed file directly for download
                def cleanup_temp_file():
                    """Clean up temporary file after sending"""
                    try:
                        if os.path.exists(processed_file_path):
                            os.remove(processed_file_path)
                            print(f"Cleaned up temporary file: {processed_file_path}")
                    except Exception as e:
                        print(f"Cleanup error: {e}")
                
                # Return the file for download with mobile-friendly headers
                response = send_file(
                    processed_file_path,
                    as_attachment=True,
                    download_name=processed_filename,
                    mimetype='video/mp4'
                )
                
                # Add mobile-friendly headers
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                
                # Schedule cleanup after response is sent
                cleanup_thread = threading.Timer(5.0, cleanup_temp_file)  # Increased delay for mobile
                cleanup_thread.start()
                
                return response
                
            except Exception as e:
                # Clean up uploaded file if processing fails
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f"Video processing error: {str(e)}")
                return jsonify({'error': f'Video processing failed: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, WebM, or 3GP files.'}), 400
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/mobile-info')
def mobile_info():
    """Mobile-specific information endpoint"""
    return jsonify({
        'mobile_optimized': True,
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024),
        'max_resolution': f"{MOBILE_MAX_RESOLUTION[0]}x{MOBILE_MAX_RESOLUTION[1]}",
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'recommendations': {
            'file_size': 'Keep videos under 100MB for best performance',
            'resolution': 'Videos will be auto-resized to mobile-friendly dimensions',
            'format': 'MP4 works best on mobile devices'
        }
    })

@app.route('/status')
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Mobile-optimized Phonk video processor is ready',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'mobile_optimized': True
    })

# @app.route('/')
# def index():
#     """Basic info endpoint"""
#     return jsonify({
#         'name': 'Mobile Phonk Video Processor API',
#         'version': '2.0.0-mobile',
#         'mobile_optimized': True,
#         'endpoints': {
#             '/upload': 'POST - Upload and process video (mobile optimized)',
#             '/mobile-info': 'GET - Mobile-specific information',
#             '/status': 'GET - Check API status'
#         }
#     })


@app.route('/')
def index():
    """Basic info endpoint"""
    return render_template('index.html', 
                       name='Mobile Phonk Video Processor API',
                       version='2.0.0-mobile',
                       mobile_optimized=True,
                       endpoints={
                           '/upload': 'POST - Upload and process video (mobile optimized)',
                           '/mobile-info': 'GET - Mobile-specific information',
                           '/status': 'GET - Check API status'
                       })

if __name__ == '__main__':
    print("Starting Mobile-Optimized Phonk Video Processor Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"Max file size: {MAX_CONTENT_LENGTH // (1024 * 1024)}MB")
    print(f"Max resolution: {MOBILE_MAX_RESOLUTION[0]}x{MOBILE_MAX_RESOLUTION[1]}")
    # print("Server running on http://localhost:5000")
    
    # Run the Flask app with mobile-friendly settings
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
