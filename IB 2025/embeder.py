import os
import shutil
import subprocess

import cv2
import numpy as np
from pydub import AudioSegment
from scipy.fftpack import fft, ifft
from scipy.io import wavfile
from tqdm import tqdm

TEMP_DIR = "temp_processing"
SEGMENT_DURATION = 1  # секунд для видео-маркировки


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cleanup():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def text_to_bits(user_id):
    return format(int(user_id), '032b')


def get_video_props(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    return duration, fps


def apply_video_wm_to_segment(input_path, output_path, user_id, strength=50):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bits = text_to_bits(user_id)
    bits_len = len(bits)

    pbar = tqdm(total=total_frames, desc=f"Video Part: {os.path.basename(input_path)}", unit="fr")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = np.float32(y)
        h, w = y.shape

        bit_idx = 0
        for i in range(0, h - h % 8, 8):
            for j in range(0, w - w % 8, 8):
                block = y[i:i + 8, j:j + 8]
                dct_block = cv2.dct(block)

                c1, c2 = dct_block[4, 3], dct_block[3, 4]
                current_bit = int(bits[bit_idx % bits_len])

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

        y = np.uint8(np.clip(y, 0, 255))
        merged = cv2.merge((y, cr, cb))
        out.write(cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR))
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()


class AudioWatermark:
    def __init__(self, user_id, id_length=32):
        self.bits = format(int(user_id), f'0{id_length}b')
        self.chunk_size = 4096
        self.bin_start, self.bin_mid, self.bin_end = 50, 60, 70
        self.silence_thresh = 500

    def embed(self, input_wav, output_wav):
        audio = AudioSegment.from_file(input_wav)
        audio = audio.set_channels(1).set_frame_rate(44100)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

        processed_samples = []
        bit_idx = 0

        for i in range(0, len(samples) - self.chunk_size, self.chunk_size):
            chunk = samples[i: i + self.chunk_size]
            if np.max(np.abs(chunk)) < self.silence_thresh:
                processed_samples.extend(chunk)
                continue

            spectrum = fft(chunk)
            current_bit = int(self.bits[bit_idx % len(self.bits)])
            idx_a, idx_b = slice(self.bin_start, self.bin_mid), slice(self.bin_mid, self.bin_end)

            if current_bit == 1:
                spectrum[idx_a] *= 2.5
                spectrum[idx_b] *= 0.4
            else:
                spectrum[idx_a] *= 0.4
                spectrum[idx_b] *= 2.5

            modified_chunk = np.clip(ifft(spectrum).real, -32768, 32767)
            processed_samples.extend(modified_chunk.astype(np.int16))
            bit_idx += 1

        wavfile.write(output_wav, 44100, np.array(processed_samples, dtype=np.int16))


def process_dual_watermark(input_video, output_video, user_id):
    ensure_dir(TEMP_DIR)

    temp_audio_ext = os.path.join(TEMP_DIR, "extracted.wav")
    temp_audio_wm = os.path.join(TEMP_DIR, "watermarked.wav")
    video_only_marked = os.path.join(TEMP_DIR, "video_marked_only.mp4")

    # 1. ОБРАБОТКА АУДИО
    print("[1/4] Извлечение и шифрование аудио...")
    subprocess.run(
        ['ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', temp_audio_ext],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    awm = AudioWatermark(user_id)
    awm.embed(temp_audio_ext, temp_audio_wm)

    # 2. ОБРАБОТКА ВИДЕО
    print("[2/4] Шифрование видео (DCT Turbo)...")
    duration, fps = get_video_props(input_video)

    p_start_src = os.path.join(TEMP_DIR, "p1_src.mp4")
    p_mid_src = os.path.join(TEMP_DIR, "p2_src.mp4")
    p_end_src = os.path.join(TEMP_DIR, "p3_src.mp4")
    p_start_wm = os.path.join(TEMP_DIR, "p1_wm.mp4")
    p_end_wm = os.path.join(TEMP_DIR, "p3_wm.mp4")

    ts1, ts2, ts3 = [os.path.join(TEMP_DIR, f"{i}.ts") for i in (1, 2, 3)]

    start_time_end = max(0, duration - SEGMENT_DURATION)

    # используем mpeg4 для промежуточных частей
    subprocess.run(['ffmpeg', '-y', '-i', input_video, '-t', str(SEGMENT_DURATION), '-c:v', 'mpeg4', '-q:v', '1', '-an',
                    p_start_src], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ['ffmpeg', '-y', '-i', input_video, '-ss', str(start_time_end), '-c:v', 'mpeg4', '-q:v', '1', '-an', p_end_src],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ['ffmpeg', '-y', '-i', input_video, '-ss', str(SEGMENT_DURATION), '-to', str(start_time_end), '-c', 'copy',
         '-an', p_mid_src], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    apply_video_wm_to_segment(p_start_src, p_start_wm, user_id)
    apply_video_wm_to_segment(p_end_src, p_end_wm, user_id)

    print("[3/4] Подготовка видео-потока (Конвертация в TS)...")

    subprocess.run(
        ['ffmpeg', '-y', '-i', p_start_wm, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-f', 'mpegts',
         ts1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ['ffmpeg', '-y', '-i', p_end_wm, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-f', 'mpegts', ts3],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # '-c copy', чтобы не ждать перекодирования всего видео ы
    # '-bsf:v h264_mp4toannexb' для совместимости форматов при склейке
    print("    ...обработка основной части видео (без перекодирования)...")
    subprocess.run(['ffmpeg', '-y', '-i', p_mid_src, '-c', 'copy', '-bsf:v', 'h264_mp4toannexb', '-f', 'mpegts', ts2],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    concat_str = f"concat:{ts1}|{ts2}|{ts3}"
    subprocess.run(['ffmpeg', '-y', '-i', concat_str, '-c', 'copy', '-an', video_only_marked],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[4/4] Финальная сборка файла...")
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_only_marked,
        '-i', temp_audio_wm,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        output_video
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cleanup()
    print(f"\n[SUCCESS] Готово! Результат: {output_video}")


if __name__ == "__main__":
    USER_ID = "34567890"
    process_dual_watermark('input1.mp4', 'output_protected.mp4', USER_ID)
