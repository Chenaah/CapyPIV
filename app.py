from flask import Flask, render_template, request, send_file, jsonify
import os
from pathlib import Path
import tempfile
import shutil
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import json

# Import your PIV processing functions
from piv_processor import process_video_piv, setup_piv_settings

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['RESULTS_FOLDER'] = Path('results')

# Create necessary folders
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'MOV', 'MP4', 'AVI'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in [ext.lower() for ext in ALLOWED_EXTENSIONS]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: MP4, AVI, MOV'}), 400
    
    try:
        # Save uploaded file with a unique ID
        import uuid
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        # Keep original extension
        ext = filename.rsplit('.', 1)[1].lower()
        video_filename = f"{job_id}.{ext}"
        video_path = app.config['UPLOAD_FOLDER'] / video_filename
        file.save(str(video_path))
        
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'filename': filename,
            'video_filename': video_filename,
            'duration': duration,
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height
        })
        
    except Exception as e:
        import traceback
        print(f"Error uploading video: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/video/<video_filename>')
def serve_video(video_filename):
    """Serve the uploaded video file for preview"""
    try:
        video_path = app.config['UPLOAD_FOLDER'] / video_filename
        if not video_path.exists():
            return "Video not found", 404
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return f"Error serving video: {str(e)}", 500

@app.route('/process', methods=['POST'])
def process_video():
    """Process the video with optional clipping"""
    try:
        data = request.get_json()
        video_filename = data.get('video_filename')
        job_id = data.get('job_id')
        original_filename = data.get('original_filename', 'video')  # Get original name
        start_time = data.get('start_time', 0)  # in seconds
        end_time = data.get('end_time')  # in seconds, None means to the end
        max_frames = data.get('max_frames')
        
        if not video_filename:
            return jsonify({'error': 'No video filename provided'}), 400
        
        video_path = app.config['UPLOAD_FOLDER'] / video_filename
        if not video_path.exists():
            return jsonify({'error': 'Video file not found'}), 404
        
        # Clip video if start_time or end_time is specified
        processing_video_path = video_path
        if start_time > 0 or end_time is not None:
            # Create clipped video
            clipped_filename = f"clipped_{video_filename}"
            clipped_path = app.config['UPLOAD_FOLDER'] / clipped_filename
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clipped_path), fourcc, fps, (width, height))
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps) if end_time is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            
            processing_video_path = clipped_path
        
        # Create a unique results folder for this processing job
        job_results_folder = app.config['RESULTS_FOLDER'] / job_id
        job_results_folder.mkdir(exist_ok=True)
        
        # Setup PIV settings
        settings = setup_piv_settings(job_results_folder)
        
        # Process video
        print(f"Processing video: {video_filename}")
        x, y, u_avg, v_avg, u_std, v_std, num_pairs = process_video_piv(
            processing_video_path, settings, max_frames=max_frames
        )
        
        print(f"Averaged PIV results from {num_pairs} frame pairs")
        
        # Calculate average speed and uncertainty
        speed_avg = np.sqrt(u_avg**2 + v_avg**2)
        speed_uncertainty = np.sqrt((u_avg*u_std)**2 + (v_avg*v_std)**2) / (speed_avg + 1e-10)
        
        # Generate plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average velocity field
        H = np.max(y)
        y_flipped = H - y
        v_flipped = -v_avg
        
        # Set black background
        ax1.set_facecolor('black')
        
        Q1 = ax1.quiver(
            x, y_flipped, u_avg, v_flipped, speed_avg,
            cmap='jet',
            scale=None,
            scale_units='xy', angles='xy',
            width=0.002
        )
        ax1.set_title(f'Average Velocity Field ({num_pairs} pairs)', color='white')
        
        # Create colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = plt.colorbar(Q1, cax=cax1)
        cbar1.set_label('Average velocity magnitude', color='white')
        cbar1.ax.tick_params(colors='white')
        
        # Plot 2: Velocity uncertainty
        ax2.set_facecolor('black')
        
        Q2 = ax2.quiver(
            x, y_flipped, u_avg, v_flipped, speed_uncertainty,
            cmap='viridis',
            scale=None,
            scale_units='xy', angles='xy',
            width=0.002
        )
        ax2.set_title('Velocity Uncertainty', color='white')
        
        # Create colorbar
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(Q2, cax=cax2)
        cbar2.set_label('Velocity uncertainty', color='white')
        cbar2.ax.tick_params(colors='white')
        
        for ax in [ax1, ax2]:
            ax.set_xlim(0, np.max(x))
            ax.set_ylim(H, 0)
            ax.set_aspect('equal')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        # Set figure background to black
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        
        # Save plot as image
        plot_path = job_results_folder / 'piv_plot.png'
        plt.savefig(str(plot_path), dpi=150, facecolor='black')
        plt.close()
        
        # Convert plot to base64 for display
        with open(plot_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Create filename from original video name (remove extension)
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
        output_filename = f"{base_name}_piv_results_{num_pairs}_pairs.txt"
        
        # Save averaged results
        output_path = job_results_folder / output_filename
        averaged_data = np.column_stack([
            x.flatten(), y.flatten(), 
            u_avg.flatten(), v_avg.flatten(),
            u_std.flatten(), v_std.flatten()
        ])
        np.savetxt(output_path, averaged_data, 
                   header='x y u_avg v_avg u_std v_std',
                   fmt='%.6f')
        
        print(f"Results saved to: {output_path}")
        
        # Clean up uploaded video and clipped video if any
        if video_path.exists():
            os.remove(str(video_path))
        if processing_video_path != video_path and processing_video_path.exists():
            os.remove(str(processing_video_path))
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'num_pairs': num_pairs,
            'plot_image': img_data,
            'download_url': f'/download/{job_id}'
        })
        
    except Exception as e:
        import traceback
        print(f"Error processing video: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>')
def download_results(job_id):
    try:
        job_results_folder = app.config['RESULTS_FOLDER'] / job_id
        
        # Find the results file (look for any .txt file with "piv_results" in the name)
        txt_files = list(job_results_folder.glob('*_piv_results_*.txt'))
        if not txt_files:
            return "Results file not found", 404
        
        results_file = txt_files[0]
        return send_file(
            results_file,
            as_attachment=True,
            download_name=results_file.name
        )
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500

@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup_results(job_id):
    try:
        job_results_folder = app.config['RESULTS_FOLDER'] / job_id
        if job_results_folder.exists():
            shutil.rmtree(job_results_folder)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5439)
