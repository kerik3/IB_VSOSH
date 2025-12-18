import cv2
import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment
import os


class DualExtractor:
    def __init__(self, id_length=32):
        self.id_length = id_length

        self.audio_chunk_size = 4096
        self.bin_start = 50
        self.bin_mid = 60
        self.bin_end = 70
        self.silence_thresh = 500

        self.video_frames_limit = 60  # Сколько кадров проверять в начале и конце

    def _extract_audio_id(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(44100)
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

            votes = [[0, 0] for _ in range(self.id_length)]
            bit_idx = 0

            for i in range(0, len(samples) - self.audio_chunk_size, self.audio_chunk_size):
                chunk = samples[i: i + self.audio_chunk_size]
                if np.max(np.abs(chunk)) < self.silence_thresh:
                    continue

                spectrum = fft(chunk)
                magnitudes = np.abs(spectrum)

                energy_a = np.mean(magnitudes[self.bin_start: self.bin_mid])
                energy_b = np.mean(magnitudes[self.bin_mid: self.bin_end])

                if energy_a + energy_b < 1: continue

                detected = 1 if energy_a > energy_b else 0
                votes[bit_idx % self.id_length][detected] += 1
                bit_idx += 1

            binary_res = "".join(["1" if v[1] > v[0] else "0" for v in votes])
            return int(binary_res, 2), binary_res
        except Exception as e:
            return None, str(e)

    def _process_video_frame(self, frame, votes):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, _, _ = cv2.split(ycrcb)
        y = np.float32(y)
        h, w = y.shape
        bit_idx = 0

        for i in range(0, h - h % 8, 8):
            for j in range(0, w - w % 8, 8):
                block = y[i:i + 8, j:j + 8]
                dct_block = cv2.dct(block)

                c1 = dct_block[4, 3]
                c2 = dct_block[3, 4]

                detected_bit = 0 if c1 > c2 else 1
                votes[bit_idx % self.id_length][detected_bit] += 1
                bit_idx += 1

    def _extract_video_id(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Не удалось открыть видео"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        votes = [[0, 0] for _ in range(self.id_length)]

        for _ in range(self.video_frames_limit):
            ret, frame = cap.read()
            if not ret: break
            self._process_video_frame(frame, votes)

        if total_frames > self.video_frames_limit * 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - self.video_frames_limit)
            while True:
                ret, frame = cap.read()
                if not ret: break
                self._process_video_frame(frame, votes)

        cap.release()
        binary_res = "".join(["1" if v[1] > v[0] else "0" for v in votes])
        try:
            return int(binary_res, 2), binary_res
        except:
            return None, "Ошибка конвертации"

    def extract_all(self, file_path):
        print(f"[*] Начинаю полный анализ файла: {file_path}")

        # 1. Видео
        v_id, v_bin = self._extract_video_id(file_path)

        # 2. Аудио
        a_id, a_bin = self._extract_audio_id(file_path)

        # Вывод результатов
        print("\n" + "=" * 40)
        print(f"{'РЕЗУЛЬТАТЫ ЭКСТРАКЦИИ':^40}")
        print("=" * 40)

        if v_id is not None:
            print(f"[VIDEO] ID: {v_id}")
            print(f"[VIDEO] Bin: {v_bin}")
        else:
            print(f"[VIDEO] Ошибка: {v_bin}")
            print(1234)

        print("-" * 40)

        if a_id is not None:
            print(f"[AUDIO] ID: {a_id}")
            print(f"[AUDIO] Bin: {a_bin}")
        else:
            print(f"[AUDIO] Ошибка: {a_bin}")

        print("=" * 40)

        if v_id == a_id and v_id is not None:
            print(f"✅ ВНИМАНИЕ: Данные совпали! Итоговый ID: {v_id}")
        elif v_id is not None and a_id is not None:
            print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: ID видео ({v_id}) и аудио ({a_id}) различаются!")

        return v_id, a_id


if __name__ == "__main__":
    FILE_TO_CHECK = 'output_protected.mp4'

    if os.path.exists(FILE_TO_CHECK):
        extractor = DualExtractor()
        extractor.extract_all(FILE_TO_CHECK)
    else:
        print(f"Файл {FILE_TO_CHECK} не найден.")