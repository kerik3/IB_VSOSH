"""
Improved Video Watermark Extraction Module
Extracts embedded user IDs from watermarked videos
"""

import os
import logging
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Custom exception for extraction errors"""
    pass


class DualExtractor:
    """Extract watermarks from both video and audio streams"""
    
    def __init__(self, id_length: int = 32):
        self.id_length = id_length
        
        # Audio parameters (must match embedder)
        self.audio_chunk_size = 4096
        self.bin_start = 50
        self.bin_mid = 60
        self.bin_end = 70
        self.silence_thresh = 500
        
        # Video parameters
        self.video_frames_limit = 60  # Frames to check at start/end
    
    def _extract_audio_id(self, file_path: str) -> Tuple[Optional[int], str]:
        """
        Extract watermark ID from audio stream
        
        Returns:
            Tuple of (extracted_id, binary_string or error_message)
        """
        try:
            logger.info("Extracting watermark from audio...")
            
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(44100)
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            
            # Voting system for robust extraction
            votes = [[0, 0] for _ in range(self.id_length)]
            bit_idx = 0
            chunks_analyzed = 0
            
            for i in range(0, len(samples) - self.audio_chunk_size, self.audio_chunk_size):
                chunk = samples[i: i + self.audio_chunk_size]
                
                # Skip silent chunks
                if np.max(np.abs(chunk)) < self.silence_thresh:
                    continue
                
                # Analyze frequency spectrum
                spectrum = fft(chunk)
                magnitudes = np.abs(spectrum)
                
                energy_a = np.mean(magnitudes[self.bin_start: self.bin_mid])
                energy_b = np.mean(magnitudes[self.bin_mid: self.bin_end])
                
                if energy_a + energy_b < 1:
                    continue
                
                # Vote for bit value based on energy distribution
                detected = 1 if energy_a > energy_b else 0
                votes[bit_idx % self.id_length][detected] += 1
                bit_idx += 1
                chunks_analyzed += 1
            
            logger.info(f"Analyzed {chunks_analyzed} audio chunks")
            
            # Determine final bits by majority voting
            binary_res = "".join(["1" if v[1] > v[0] else "0" for v in votes])
            
            # Calculate confidence score
            confidence_scores = []
            for v in votes:
                total = sum(v)
                if total > 0:
                    confidence = max(v) / total
                    confidence_scores.append(confidence)
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            logger.info(f"Audio extraction confidence: {avg_confidence:.2%}")
            
            extracted_id = int(binary_res, 2)
            return extracted_id, binary_res
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None, str(e)
    
    def _process_video_frame(self, frame: np.ndarray, votes: list) -> None:
        """
        Process a single video frame to extract watermark bits
        
        Args:
            frame: Video frame in BGR format
            votes: Voting array to accumulate bit detections
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, _, _ = cv2.split(ycrcb)
        y = np.float32(y)
        h, w = y.shape
        bit_idx = 0
        
        # Process 8x8 blocks
        for i in range(0, h - h % 8, 8):
            for j in range(0, w - w % 8, 8):
                block = y[i:i + 8, j:j + 8]
                dct_block = cv2.dct(block)
                
                # Extract from mid-frequency coefficients
                c1 = dct_block[4, 3]
                c2 = dct_block[3, 4]
                
                # Detect embedded bit
                detected_bit = 0 if c1 > c2 else 1
                votes[bit_idx % self.id_length][detected_bit] += 1
                bit_idx += 1
    
    def _extract_video_id(self, video_path: str) -> Tuple[Optional[int], str]:
        """
        Extract watermark ID from video stream
        
        Returns:
            Tuple of (extracted_id, binary_string or error_message)
        """
        try:
            logger.info("Extracting watermark from video...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, "Failed to open video file"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video: {total_frames} frames @ {fps} fps")
            
            # Voting system for robust extraction
            votes = [[0, 0] for _ in range(self.id_length)]
            frames_analyzed = 0
            
            # Process frames from the beginning
            for _ in range(min(self.video_frames_limit, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                self._process_video_frame(frame, votes)
                frames_analyzed += 1
            
            # Process frames from the end
            if total_frames > self.video_frames_limit * 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - self.video_frames_limit)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self._process_video_frame(frame, votes)
                    frames_analyzed += 1
            
            cap.release()
            
            logger.info(f"Analyzed {frames_analyzed} video frames")
            
            # Determine final bits by majority voting
            binary_res = "".join(["1" if v[1] > v[0] else "0" for v in votes])
            
            # Calculate confidence score
            confidence_scores = []
            for v in votes:
                total = sum(v)
                if total > 0:
                    confidence = max(v) / total
                    confidence_scores.append(confidence)
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            logger.info(f"Video extraction confidence: {avg_confidence:.2%}")
            
            try:
                extracted_id = int(binary_res, 2)
                return extracted_id, binary_res
            except ValueError:
                return None, "Failed to convert binary to integer"
                
        except Exception as e:
            logger.error(f"Video extraction failed: {e}")
            return None, str(e)
    
    def extract_all(self, file_path: str) -> Dict:
        """
        Extract watermarks from both video and audio streams
        
        Args:
            file_path: Path to watermarked video file
            
        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(file_path):
            raise ExtractionError(f"File not found: {file_path}")
        
        logger.info(f"Starting full extraction analysis: {file_path}")
        logger.info("=" * 60)
        
        # Extract from video
        v_id, v_bin = self._extract_video_id(file_path)
        
        # Extract from audio
        a_id, a_bin = self._extract_audio_id(file_path)
        
        # Prepare results
        results = {
            'file_path': file_path,
            'video': {
                'id': v_id,
                'binary': v_bin if v_id is not None else None,
                'success': v_id is not None,
                'error': v_bin if v_id is None else None
            },
            'audio': {
                'id': a_id,
                'binary': a_bin if a_id is not None else None,
                'success': a_id is not None,
                'error': a_bin if a_id is None else None
            },
            'match': False,
            'confidence': 'none'
        }
        
        # Check consistency
        if v_id is not None and a_id is not None:
            if v_id == a_id:
                results['match'] = True
                results['confidence'] = 'high'
                results['final_id'] = v_id
                logger.info(f"✅ SUCCESS: Both streams match! ID: {v_id}")
            else:
                results['match'] = False
                results['confidence'] = 'low'
                logger.warning(f"⚠️  WARNING: Mismatch! Video ID: {v_id}, Audio ID: {a_id}")
        elif v_id is not None:
            results['final_id'] = v_id
            results['confidence'] = 'medium'
            logger.info(f"ℹ️  Partial: Only video ID extracted: {v_id}")
        elif a_id is not None:
            results['final_id'] = a_id
            results['confidence'] = 'medium'
            logger.info(f"ℹ️  Partial: Only audio ID extracted: {a_id}")
        else:
            logger.error("❌ FAILED: Could not extract watermark from either stream")
        
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict) -> None:
        """Pretty print extraction results"""
        print("\n" + "=" * 60)
        print(f"{'WATERMARK EXTRACTION RESULTS':^60}")
        print("=" * 60)
        
        # Video results
        if results['video']['success']:
            print(f"[VIDEO] ID: {results['video']['id']}")
            print(f"[VIDEO] Binary: {results['video']['binary']}")
        else:
            print(f"[VIDEO] Error: {results['video']['error']}")
        
        print("-" * 60)
        
        # Audio results
        if results['audio']['success']:
            print(f"[AUDIO] ID: {results['audio']['id']}")
            print(f"[AUDIO] Binary: {results['audio']['binary']}")
        else:
            print(f"[AUDIO] Error: {results['audio']['error']}")
        
        print("=" * 60)
        
        # Final verdict
        if results['match']:
            print(f"✅ MATCH CONFIRMED - Final ID: {results['final_id']}")
            print(f"Confidence: {results['confidence'].upper()}")
        elif 'final_id' in results:
            print(f"⚠️  PARTIAL MATCH - ID: {results['final_id']}")
            print(f"Confidence: {results['confidence'].upper()}")
        else:
            print("❌ EXTRACTION FAILED")
        
        print("=" * 60 + "\n")


def extract_watermark(file_path: str, id_length: int = 32) -> Dict:
    """
    Convenience function to extract watermark from a file
    
    Args:
        file_path: Path to watermarked video
        id_length: Expected watermark bit length
        
    Returns:
        Dictionary with extraction results
    """
    extractor = DualExtractor(id_length=id_length)
    return extractor.extract_all(file_path)


if __name__ == "__main__":
    import sys
    
    FILE_TO_CHECK = 'output_protected.mp4'
    
    if len(sys.argv) > 1:
        FILE_TO_CHECK = sys.argv[1]
    
    if os.path.exists(FILE_TO_CHECK):
        try:
            result = extract_watermark(FILE_TO_CHECK)
            
            # Exit with appropriate code
            if result['match']:
                sys.exit(0)  # Perfect match
            elif 'final_id' in result:
                sys.exit(1)  # Partial match
            else:
                sys.exit(2)  # Failed
                
        except ExtractionError as e:
            logger.error(f"Extraction error: {e}")
            sys.exit(3)
    else:
        logger.error(f"File not found: {FILE_TO_CHECK}")
        sys.exit(4)
