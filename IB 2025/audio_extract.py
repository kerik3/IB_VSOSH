import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment


class AudioExtractor:
    def __init__(self, id_length=32):
        self.id_length = id_length
        self.chunk_size = 4096
        self.bin_start = 50
        self.bin_mid = 60
        self.bin_end = 70
        self.silence_thresh = 500

    def extract(self, file_path):
        print(f"[*] Анализ аудио: {file_path}")

        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(44100)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

        votes = [[0, 0] for _ in range(self.id_length)]
        bit_idx = 0
        valid_chunks = 0

        for i in range(0, len(samples) - self.chunk_size, self.chunk_size):
            chunk = samples[i: i + self.chunk_size]

            # Пропускаем тишину (она дает рандомный результат)
            if np.max(np.abs(chunk)) < self.silence_thresh:
                continue

            spectrum = fft(chunk)
            magnitudes = np.abs(spectrum)

            # Считаем среднюю энергию
            energy_a = np.mean(magnitudes[self.bin_start: self.bin_mid])
            energy_b = np.mean(magnitudes[self.bin_mid: self.bin_end])

            # Защита от деления на ноль, если блок странный
            if energy_a + energy_b < 1:
                continue

            # Определяем бит
            if energy_a > energy_b:
                detected = 1
            else:
                detected = 0

            votes[bit_idx % self.id_length][detected] += 1
            bit_idx += 1
            valid_chunks += 1

        print(f"[*] Проанализировано активных блоков: {valid_chunks}")

        # Сборка результата
        binary_res = ""
        for v0, v1 in votes:
            # Если голосов мало, результат может быть ненадежным
            binary_res += "1" if v1 > v0 else "0"

        try:
            print("\n" + "=" * 30)
            print(f"БИНАРНЫЙ КОД: {binary_res}")
            print(f"РАСШИФРОВАННЫЙ ID: {int(binary_res, 2)}")
            print("=" * 30)
        except:
            print("Ошибка конвертации.")

# --- ЗАПУСК ---
ex = AudioExtractor()
ex.extract('PROTECTED_1111.mp4')