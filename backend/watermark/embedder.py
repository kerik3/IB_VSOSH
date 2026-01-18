"""
Improved Video Watermarking Module
Embeds unique user IDs into both video and audio streams using DCT and FFT
"""

import os
import shutil
import subprocess
import logging
from typing import Optional, Tuple
import hashlib

import cv2
import numpy as np
from pydub import AudioSegment
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatermarkConfig:
    """Configuration for watermarking parameters"""
    TEMP_DIR = "temp_processing"
    SEGMENT_DURATION = 1  # seconds for video marking
    VIDEO_STRENGTH = 50  # DCT modification strength
    AUDIO_CHUNK_SIZE = 4096
    AUDIO_BIN_START = 50
    AUDIO_BIN_MID = 60
    AUDIO_BIN_END = 70
    SILENCE_THRESHOLD = 500
    ID_LENGTH = 32  # bits


class VideoWatermarkError(Exception):
    """Custom exception for watermarking errors"""
    pass


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def cleanup(temp_dir: str = WatermarkConfig.TEMP_DIR) -> None:
    """Remove temporary processing directory"""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {temp_dir}: {e}")


def generate_secure_id(user_id: str, video_id: str) -> str:
    """
    Generate a secure watermark ID by hashing user_id and video_id
    This makes it harder to guess or forge watermarks
    """
    combined = f"{user_id}:{video_id}:{os.urandom(8).hex()}"
    hash_obj = hashlib.sha256(combined.encode())
    # Use first 32 bits of hash as numeric ID
    return str(int(hash_obj.hexdigest()[:8], 16))


def text_to_bits(user_id: str, bit_length: int = WatermarkConfig.ID_LENGTH) -> str:
    """Convert user ID to binary string with specified length"""
    try:
        numeric_id = int(user_id)
        return format(numeric_id, f'0{bit_length}b')
    except ValueError:
        raise VideoWatermarkError(f"Invalid user ID format: {user_id}")


def get_video_properties(path: str) -> Tuple[float, float, int, int]:
    """
    Get video properties
    Returns: (duration, fps, width, height)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoWatermarkError(f"Cannot open video file: {path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0
    
    cap.release()
    
    if duration == 0:
        raise VideoWatermarkError("Video duration is 0")
    
    logger.info(f"Video properties: {width}x{height}, {fps}fps, {duration:.2f}s")
    return duration, fps, width, height


def apply_video_watermark_to_segment(
    input_path: str, 
    output_path: str, 
    user_id: str, 
    strength: int = WatermarkConfig.VIDEO_STRENGTH
) -> None:
    """
    Apply DCT-based watermark to video segment
    Embeds watermark in Y channel of YCrCb color space
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise VideoWatermarkError(f"Cannot open video: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    bits = text_to_bits(user_id)
    bits_len = len(bits)
    
    pbar = tqdm(
        total=total_frames, 
        desc=f"Watermarking: {os.path.basename(input_path)}", 
        unit="frame"
    )
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y = np.float32(y)
            h, w = y.shape
            
            # Apply DCT watermarking to 8x8 blocks
            bit_idx = 0
            for i in range(0, h - h % 8, 8):
                for j in range(0, w - w % 8, 8):
                    block = y[i:i + 8, j:j + 8]
                    dct_block = cv2.dct(block)
                    
                    # Modify mid-frequency coefficients
                    c1, c2 = dct_block[4, 3], dct_block[3, 4]
                    current_bit = int(bits[bit_idx % bits_len])
                    
                    # Embed bit by modifying coefficient relationship
                    if current_bit == 0:
                        if c1 <= c2 + strength:
                            diff = (c2 + strength - c1) / 2.0
                            c1 += diff
                            c2 -= diff
                    else:
                        if c2 <= c1 + strength:
                            diff = (c1 + strength - c2) / 2.0
                            c2 += diff
                            c1 -= diff
                    
                    dct_block[4, 3], dct_block[3, 4] = c1, c2
                    y[i:i + 8, j:j + 8] = cv2.idct(dct_block)
                    bit_idx += 1
            
            # Convert back to BGR
            y = np.uint8(np.clip(y, 0, 255))
            merged = cv2.merge((y, cr, cb))
            frame_out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            out.write(frame_out)
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")
            raise VideoWatermarkError(f"Frame processing failed: {e}")
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    logger.info(f"Processed {frame_count} frames")


class AudioWatermark:
    """Audio watermarking using FFT-based frequency domain manipulation"""
    
    def __init__(self, user_id: str, id_length: int = WatermarkConfig.ID_LENGTH):
        self.bits = text_to_bits(user_id, id_length)
        self.chunk_size = WatermarkConfig.AUDIO_CHUNK_SIZE
        self.bin_start = WatermarkConfig.AUDIO_BIN_START
        self.bin_mid = WatermarkConfig.AUDIO_BIN_MID
        self.bin_end = WatermarkConfig.AUDIO_BIN_END
        self.silence_thresh = WatermarkConfig.SILENCE_THRESHOLD
    
    def embed(self, input_wav: str, output_wav: str) -> None:
        """Embed watermark into audio file"""
        try:
            audio = AudioSegment.from_file(input_wav)
            audio = audio.set_channels(1).set_frame_rate(44100)
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            
            logger.info(f"Processing audio: {len(samples)} samples")
            
            # Use a copy for output to avoid list overhead (memory optimization)
            output_samples = samples.copy()
            
            bit_idx = 0
            chunks_processed = 0
            
            # Process chunks
            for i in range(0, len(samples) - self.chunk_size, self.chunk_size):
                chunk = samples[i: i + self.chunk_size]
                
                # Skip silent chunks
                if np.max(np.abs(chunk)) < self.silence_thresh:
                    continue
                
                # FFT: convert to frequency domain
                spectrum = fft(chunk)
                current_bit = int(self.bits[bit_idx % len(self.bits)])
                
                # Define frequency ranges
                idx_a = slice(self.bin_start, self.bin_mid)
                idx_b = slice(self.bin_mid, self.bin_end)
                
                # Embed bit by amplifying/attenuating frequency ranges
                if current_bit == 1:
                    spectrum[idx_a] *= 2.5  # Amplify low range
                    spectrum[idx_b] *= 0.4  # Attenuate high range
                else:
                    spectrum[idx_a] *= 0.4  # Attenuate low range
                    spectrum[idx_b] *= 2.5  # Amplify high range
                
                # IFFT: convert back to time domain
                modified_chunk = np.clip(ifft(spectrum).real, -32768, 32767)
                output_samples[i: i + self.chunk_size] = modified_chunk.astype(np.int16)
                
                bit_idx += 1
                chunks_processed += 1
            
            logger.info(f"Processed {chunks_processed} audio chunks")
            
            # Write output
            wavfile.write(output_wav, 44100, output_samples)
            
        except Exception as e:
            raise VideoWatermarkError(f"Audio watermarking failed: {e}")


def run_ffmpeg(cmd: list, description: str = "") -> None:
    """Run ffmpeg command with error handling"""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown error"
        logger.error(f"FFmpeg error during {description}: {error_msg}")
        raise VideoWatermarkError(f"FFmpeg failed: {description}")


def process_dual_watermark(
    input_video: str, 
    output_video: str, 
    user_id: str,
    video_id: str = None,
    progress_callback=None
) -> dict:
    """
    Apply dual watermark (video + audio) to a video file
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        user_id: Unique user identifier
        video_id: Video identifier for enhanced security
        progress_callback: Optional callback function for progress updates
    
    Returns:
        dict with watermark metadata
    """
    logger.info(f"Starting watermarking process for user {user_id}")
    
    # Validate input
    if not os.path.exists(input_video):
        raise VideoWatermarkError(f"Input video not found: {input_video}")
    
    # Use user_id directly as watermark_id for transparency
    watermark_id = user_id
    logger.info(f"Using watermark ID: {watermark_id}")
    
    try:
        ensure_dir(WatermarkConfig.TEMP_DIR)
        
        temp_audio_ext = os.path.join(WatermarkConfig.TEMP_DIR, "extracted.wav")
        temp_audio_wm = os.path.join(WatermarkConfig.TEMP_DIR, "watermarked.wav")
        video_only_marked = os.path.join(WatermarkConfig.TEMP_DIR, "video_marked_only.mp4")
        
        # STEP 1: Audio Processing
        logger.info("[1/4] Extracting and watermarking audio...")
        if progress_callback:
            progress_callback(10, "Extracting audio")
        
        run_ffmpeg([
            'ffmpeg', '-y', '-i', input_video,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
            temp_audio_ext
        ], "audio extraction")
        
        awm = AudioWatermark(watermark_id)
        awm.embed(temp_audio_ext, temp_audio_wm)
        
        # STEP 2: Video Processing
        logger.info("[2/4] Watermarking video (DCT method)...")
        if progress_callback:
            progress_callback(30, "Watermarking video")
        
        duration, fps, width, height = get_video_properties(input_video)
        
        # Define temporary file paths
        p_start_src = os.path.join(WatermarkConfig.TEMP_DIR, "p1_src.mp4")
        p_mid_src = os.path.join(WatermarkConfig.TEMP_DIR, "p2_src.mp4")
        p_end_src = os.path.join(WatermarkConfig.TEMP_DIR, "p3_src.mp4")
        p_start_wm = os.path.join(WatermarkConfig.TEMP_DIR, "p1_wm.mp4")
        p_end_wm = os.path.join(WatermarkConfig.TEMP_DIR, "p3_wm.mp4")
        
        ts1, ts2, ts3 = [os.path.join(WatermarkConfig.TEMP_DIR, f"{i}.ts") for i in (1, 2, 3)]
        
        start_time_end = max(0, duration - WatermarkConfig.SEGMENT_DURATION)
        
        # Extract segments
        run_ffmpeg([
            'ffmpeg', '-y', '-i', input_video,
            '-t', str(WatermarkConfig.SEGMENT_DURATION),
            '-c:v', 'mpeg4', '-q:v', '1', '-an', p_start_src
        ], "start segment extraction")
        
        run_ffmpeg([
            'ffmpeg', '-y', '-i', input_video,
            '-ss', str(start_time_end),
            '-c:v', 'mpeg4', '-q:v', '1', '-an', p_end_src
        ], "end segment extraction")
        
        run_ffmpeg([
            'ffmpeg', '-y', '-i', input_video,
            '-ss', str(WatermarkConfig.SEGMENT_DURATION),
            '-to', str(start_time_end),
            '-c', 'copy', '-an', p_mid_src
        ], "middle segment extraction")
        
        # Apply watermark to start and end segments
        apply_video_watermark_to_segment(p_start_src, p_start_wm, watermark_id)
        apply_video_watermark_to_segment(p_end_src, p_end_wm, watermark_id)
        
        # STEP 3: Prepare video stream
        logger.info("[3/4] Preparing video stream (converting to TS)...")
        if progress_callback:
            progress_callback(60, "Encoding video")
        
        run_ffmpeg([
            'ffmpeg', '-y', '-i', p_start_wm,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            '-f', 'mpegts', ts1
        ], "TS conversion part 1")
        
        run_ffmpeg([
            'ffmpeg', '-y', '-i', p_end_wm,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            '-f', 'mpegts', ts3
        ], "TS conversion part 3")
        
        run_ffmpeg([
            'ffmpeg', '-y', '-i', p_mid_src,
            '-c', 'copy', '-bsf:v', 'h264_mp4toannexb',
            '-f', 'mpegts', ts2
        ], "TS conversion part 2")
        
        # Concatenate segments
        concat_str = f"concat:{ts1}|{ts2}|{ts3}"
        run_ffmpeg([
            'ffmpeg', '-y', '-i', concat_str,
            '-c', 'copy', '-an', video_only_marked
        ], "segment concatenation")
        
        # STEP 4: Final assembly
        logger.info("[4/4] Final assembly...")
        if progress_callback:
            progress_callback(90, "Final assembly")
        
        run_ffmpeg([
            'ffmpeg', '-y',
            '-i', video_only_marked,
            '-i', temp_audio_wm,
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            output_video
        ], "final assembly")
        
        cleanup()
        
        logger.info(f"[SUCCESS] Watermarked video created: {output_video}")
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            'success': True,
            'watermark_id': watermark_id,
            'user_id': user_id,
            'video_id': video_id,
            'output_path': output_video,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }
        
    except Exception as e:
        cleanup()
        logger.error(f"Watermarking failed: {e}")
        raise VideoWatermarkError(f"Watermarking process failed: {str(e)}")


if __name__ == "__main__":
    # Test example
    USER_ID = "12345678"
    VIDEO_ID = "video_001"
    
    try:
        result = process_dual_watermark(
            'test_input.mp4',
            'output_protected.mp4',
            USER_ID,
            VIDEO_ID
        )
        print(f"Success! Watermark info: {result}")
    except VideoWatermarkError as e:
        print(f"Error: {e}")
