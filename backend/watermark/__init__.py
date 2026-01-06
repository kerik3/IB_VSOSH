"""
VVM Watermarking Module
Dual watermarking system for video protection
"""

from .embedder import (
    process_dual_watermark,
    VideoWatermarkError,
    WatermarkConfig,
    generate_secure_id
)
from .extractor import (
    extract_watermark,
    DualExtractor,
    ExtractionError
)

__all__ = [
    'process_dual_watermark',
    'extract_watermark',
    'VideoWatermarkError',
    'ExtractionError',
    'WatermarkConfig',
    'DualExtractor',
    'generate_secure_id'
]
