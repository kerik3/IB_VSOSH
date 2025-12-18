import cv2
import numpy as np
import subprocess
import os
import shutil
from tqdm import tqdm

TEMP_DIR = "temp_parts"
SEGMENT_DURATION = 5  # 5 секунд в начале и конце


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


def apply_watermark_to_segment(input_path, output_path, user_id, strength=50):
    # STRENGTH = 50 (Было 20). Для Turbo режима нужно сильнее,
    # так как двойная конвертация убивает слабый сигнал.

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bits = text_to_bits(user_id)
    bits_len = len(bits)

    pbar = tqdm(total=total_frames, desc=f"Code part: {os.path.basename(input_path)}", unit="fr")

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

                c1 = dct_block[4, 3]
                c2 = dct_block[3, 4]
                current_bit = int(bits[bit_idx % bits_len])

                # Усиленная логика
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

                dct_block[4, 3] = c1
                dct_block[3, 4] = c2
                y[i:i + 8, j:j + 8] = cv2.idct(dct_block)
                bit_idx += 1

        y = np.uint8(np.clip(y, 0, 255))
        merged = cv2.merge((y, cr, cb))
        final_frame = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        out.write(final_frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()


def embed_turbo(video_path, output_path, user_id):
    ensure_dir(TEMP_DIR)

    duration, fps = get_video_props(video_path)
    print(f"[*] Длительность: {duration:.2f} сек")

    if duration < SEGMENT_DURATION * 3:
        print("[-] Видео слишком короткое.")
        return

    # Пути
    p_start_src = os.path.join(TEMP_DIR, "part1_src.mp4")
    p_mid_src = os.path.join(TEMP_DIR, "part2_src.mp4")
    p_end_src = os.path.join(TEMP_DIR, "part3_src.mp4")

    p_start_wm = os.path.join(TEMP_DIR, "part1_wm.mp4")
    p_end_wm = os.path.join(TEMP_DIR, "part3_wm.mp4")

    ts_start = os.path.join(TEMP_DIR, "1.ts")
    ts_mid = os.path.join(TEMP_DIR, "2.ts")
    ts_end = os.path.join(TEMP_DIR, "3.ts")

    print("[1/5] Нарезка (High Quality)...")
    # Добавлен флаг -q:v 1 (максимальное качество для mpeg4)
    # Нарезка
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-t', str(SEGMENT_DURATION), '-c:v', 'mpeg4', '-q:v', '1', p_start_src],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    start_time_end = duration - SEGMENT_DURATION
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-ss', str(start_time_end), '-c:v', 'mpeg4', '-q:v', '1', p_end_src],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Середина копируется
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-ss', str(SEGMENT_DURATION), '-to', str(start_time_end), '-c', 'copy',
         p_mid_src], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[2/5] Встраивание ID...")
    apply_watermark_to_segment(p_start_src, p_start_wm, user_id, strength=50)  # Сила 50
    apply_watermark_to_segment(p_end_src, p_end_wm, user_id, strength=50)  # Сила 50

    print("[3/5] Конвертация в TS...")
    # Конвертируем в TS с высоким битрейтом, чтобы не потерять метку перед склейкой
    subprocess.run(['ffmpeg', '-y', '-i', p_start_wm, '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-bsf:v',
                    'h264_mp4toannexb', '-f', 'mpegts', ts_start], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ['ffmpeg', '-y', '-i', p_mid_src, '-c', 'copy', '-bsf:v', 'h264_mp4toannexb', '-f', 'mpegts', ts_mid],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-y', '-i', p_end_wm, '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-bsf:v',
                    'h264_mp4toannexb', '-f', 'mpegts', ts_end], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[4/5] Финальная склейка...")
    concat_string = f"concat:{ts_start}|{ts_mid}|{ts_end}"
    subprocess.run([
        'ffmpeg', '-y', '-i', concat_string,
        '-c', 'copy', '-bsf:a', 'aac_adtstoasc',
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cleanup()
    print(f"[SUCCESS] {output_path}")

# --- ЗАПУСК ---
embed_turbo('input1.mp4', 'output_turbo.mp4', 22225555)