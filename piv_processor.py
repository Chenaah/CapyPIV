"""
PIV processing module extracted from anothertrial.py
"""
from openpiv import windef, tools, scaling, validation, filters, preprocess
import openpiv.pyprocess as process
from openpiv import pyprocess
import numpy as np
import pathlib
import warnings
import cv2
from pathlib import Path
import tempfile
import shutil

def setup_piv_settings(output_folder):
    """Setup PIV settings with default parameters"""
    settings = windef.PIVSettings()
    
    # Data related settings
    settings.filepath_images = pathlib.Path('.')
    settings.save_path = pathlib.Path(output_folder)
    settings.save_folder_suffix = 'Test_110'
    settings.frame_pattern_a = 'B001_1.tif'
    settings.frame_pattern_b = 'B001_2.tif'
    
    # Region of interest
    settings.ROI = 'full'
    
    # Image preprocessing
    settings.dynamic_masking_method = 'edges'
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7
    settings.deformation_method = 'symmetric'
    
    # Processing Parameters
    settings.correlation_method = 'circular'
    settings.normalized_correlation = True
    settings.num_iterations = 3
    settings.windowsizes = (128, 64, 32)
    settings.overlap = (64, 32, 16)
    settings.subpixel_method = 'gaussian'
    settings.interpolation_order = 3
    settings.scaling_factor = 1
    settings.dt = 1
    
    # Signal to noise ratio options
    settings.extract_sig2noise = True
    settings.sig2noise_method = 'peak2mean'
    settings.sig2noise_mask = 2
    
    # Validation Parameters
    settings.validation_first_pass = True
    settings.MinMax_U_disp = (-50, 50)
    settings.MinMax_V_disp = (-50, 50)
    settings.std_threshold = 10
    settings.median_threshold = 4
    settings.median_size = 3
    settings.sig2noise_threshold = 1.2
    
    # Outlier replacement or Smoothing options
    settings.replace_vectors = True
    settings.smoothn = False
    settings.smoothn_p = 0.5
    settings.filter_method = 'localmean'
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2
    
    # Output options
    settings.save_plot = False
    settings.show_plot = False
    settings.scale_plot = 200
    settings.show_all_plots = False
    
    return settings

def extract_frames_from_video(video_path, output_dir, max_frames=None):
    """Extract frames from video and save as individual images
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract (None for all frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_count = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Stop if we've reached the maximum number of frames
        if max_frames is not None and frame_count >= max_frames:
            break
            
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_filename = f"frame_{frame_count:04d}.tif"
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    return frame_paths

def process_video_piv(video_path, settings, max_frames=None):
    """Process video frames and calculate average PIV
    
    Args:
        video_path: Path to the video file
        settings: PIV settings object
        max_frames: Maximum number of frames to process (None for all frames)
    
    Returns:
        x, y, u_avg, v_avg, u_std, v_std, valid_pairs
    """
    
    # Create temporary directory for frames
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Extract frames from video
        print(f"Extracting {'first ' + str(max_frames) if max_frames else 'all'} frames from video...")
        frame_paths = extract_frames_from_video(video_path, temp_dir, max_frames)
        
        if len(frame_paths) < 2:
            raise ValueError("Video must have at least 2 frames")
        
        print(f"Extracted {len(frame_paths)} frames")
        
        # Process consecutive frame pairs
        all_u = []
        all_v = []
        all_x = []
        all_y = []
        valid_pairs = 0
        
        for i in range(len(frame_paths) - 1):
            print(f"Processing frame pair {i+1}/{len(frame_paths)-1}")
            
            # Update settings for current frame pair
            frame_a = frame_paths[i].name
            frame_b = frame_paths[i+1].name
            
            # Temporarily update settings
            original_path = settings.filepath_images
            original_pattern_a = settings.frame_pattern_a
            original_pattern_b = settings.frame_pattern_b
            original_save_folder = settings.save_folder_suffix
            
            settings.filepath_images = temp_dir
            settings.frame_pattern_a = frame_a
            settings.frame_pattern_b = frame_b
            settings.save_folder_suffix = f'pair_{i:04d}'
            
            try:
                # Run PIV for this frame pair
                windef.piv(settings)
                
                # Load results
                result_file = settings.save_path / f"OpenPIV_results_{settings.windowsizes[-1]}_{settings.save_folder_suffix}" / "field_A0000.txt"
                
                if result_file.exists():
                    arr = np.loadtxt(str(result_file))
                    
                    if valid_pairs == 0:
                        # Store grid coordinates from first valid pair
                        all_x = arr[:, 0]
                        all_y = arr[:, 1]
                        x_shape = len(np.unique(all_x))
                        y_shape = len(np.unique(all_y))
                    
                    all_u.append(arr[:, 2])
                    all_v.append(arr[:, 3])
                    valid_pairs += 1
                    
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
            
            finally:
                # Restore original settings
                settings.filepath_images = original_path
                settings.frame_pattern_a = original_pattern_a
                settings.frame_pattern_b = original_pattern_b
                settings.save_folder_suffix = original_save_folder
        
        if valid_pairs == 0:
            raise ValueError("No valid frame pairs were processed")
        
        print(f"Successfully processed {valid_pairs} frame pairs")
        
        # Calculate average velocity fields
        all_u = np.array(all_u)
        all_v = np.array(all_v)
        
        avg_u = np.mean(all_u, axis=0)
        avg_v = np.mean(all_v, axis=0)
        std_u = np.std(all_u, axis=0)
        std_v = np.std(all_v, axis=0)
        
        # Reshape back to 2D grids
        x = all_x.reshape(y_shape, x_shape)
        y = all_y.reshape(y_shape, x_shape)
        u_avg = avg_u.reshape(y_shape, x_shape)
        v_avg = avg_v.reshape(y_shape, x_shape)
        u_std = std_u.reshape(y_shape, x_shape)
        v_std = std_v.reshape(y_shape, x_shape)
        
        return x, y, u_avg, v_avg, u_std, v_std, valid_pairs
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)
