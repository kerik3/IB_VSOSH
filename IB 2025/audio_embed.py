import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from pydub import AudioSegment
import os
import subprocess


class AudioWatermark:
    def __init__(self, user_id, id_length=32):
        self.bits = format(int(user_id), f'0{id_length}b')
        self.chunk_size = 4096  # Увеличили размер окна для точности

        # --- ВАЖНОЕ ИЗМЕНЕНИЕ: СОСЕДНИЕ ДИАПАЗОНЫ ---
        # Чтобы избежать ошибки "все единицы", берем частоты рядом
        # При 44100 Гц и окне 4096, 1 бин ~ 10 Гц.
        # Берем диапазон ~500-600 Гц vs ~600-700 Гц
        self.bin_start = 50
        self.bin_mid = 60
        self.bin_end = 70

        self.silence_thresh = 500  # Порог тишины (амплитуда)

    def embed(self, input_audio_path, output_audio_path):
        print(f"[*] Обработка звука из: {input_audio_path}")

        audio = AudioSegment.from_file(input_audio_path)
        audio = audio.set_channels(1).set_frame_rate(44100)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

        processed_samples = []
        bit_idx = 0

        for i in range(0, len(samples) - self.chunk_size, self.chunk_size):
            chunk = samples[i: i + self.chunk_size]

            # Если кусок тихий - не трогаем его, чтобы не портить и не ломать логику
            if np.max(np.abs(chunk)) < self.silence_thresh:
                processed_samples.extend(chunk)
                continue

            spectrum = fft(chunk)

            current_bit = int(self.bits[bit_idx % len(self.bits)])

            idx_a = slice(self.bin_start, self.bin_mid)
            idx_b = slice(self.bin_mid, self.bin_end)

            # Усиленные коэффициенты (было 1.5 и 0.8, стало жестче)
            if current_bit == 1:
                spectrum[idx_a] *= 2.5
                spectrum[idx_b] *= 0.4
            else:
                spectrum[idx_a] *= 0.4
                spectrum[idx_b] *= 2.5

            modified_chunk = ifft(spectrum).real

            # Защита от клиппинга (перегрузки звука)
            modified_chunk = np.clip(modified_chunk, -32768, 32767)

            processed_samples.extend(modified_chunk.astype(np.int16))
            bit_idx += 1

        processed_samples = np.array(processed_samples, dtype=np.int16)
        wavfile.write(output_audio_path, 44100, processed_samples)


def embed_audio_fixed(video_path, user_id):
    temp_audio = "temp_extracted.wav"
    watermarked_audio = "temp_watermarked.wav"
    final_output = f"PROTECTED_1111.mp4"

    # 1. Достаем wav
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', temp_audio],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Встраиваем
    wm = AudioWatermark(user_id)
    wm.embed(temp_audio, watermarked_audio)

    # 3. Собираем
    print(f"[*] Сборка видео {final_output}...")
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path, '-i', watermarked_audio,
        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        final_output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_audio): os.remove(temp_audio)
    if os.path.exists(watermarked_audio): os.remove(watermarked_audio)
    print(f"[SUCCESS] Готово!")

# --- ЗАПУСК ---
embed_audio_fixed('input1.mp4', 56781234)