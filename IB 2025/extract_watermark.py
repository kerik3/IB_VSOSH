import cv2
import numpy as np


def process_frame(frame, votes, id_length):
    """Вспомогательная функция для анализа одного кадра"""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
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

            votes[bit_idx % id_length][detected_bit] += 1
            bit_idx += 1


def extract_watermark_fast(video_path, id_length=32, frames_limit=60):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Анализ видео. Всего кадров: {total_frames}")

    votes = [[0, 0] for _ in range(id_length)]

    # 1. Анализируем НАЧАЛО (первые N кадров)
    print(f"[*] Сканирование начала (кадры 0-{frames_limit})...")
    for i in range(frames_limit):
        ret, frame = cap.read()
        if not ret: break
        process_frame(frame, votes, id_length)

    # 2. Прыжок в КОНЕЦ
    start_end_segment = total_frames - frames_limit
    if start_end_segment > frames_limit:  # Если видео достаточно длинное для прыжка
        print(f"[*] Перемотка к кадру {start_end_segment}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_end_segment)

        print(f"[*] Сканирование конца (кадры {start_end_segment}-{total_frames})...")
        while True:
            ret, frame = cap.read()
            if not ret: break
            process_frame(frame, votes, id_length)

    cap.release()

    # Расшифровка
    binary_res = ""
    for v0, v1 in votes:
        binary_res += "1" if v1 > v0 else "0"

    try:
        recovered_id = int(binary_res, 2)
        print("\n" + "=" * 30)
        print(f"БИНАРНЫЙ КОД: {binary_res}")
        print(f"НАЙДЕННЫЙ ID: {recovered_id}")
        print("=" * 30)
    except ValueError:
        print("Ошибка данных.")

extract_watermark_fast('output_turbo.mp4')