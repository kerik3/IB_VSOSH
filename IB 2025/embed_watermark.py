import cv2
import numpy as np
import subprocess
import os


def text_to_bits(user_id):
    return format(int(user_id), '032b')


def add_audio_to_video(original_video, watermarked_video_silent, final_output):

    print(f"[*] Объединение видео и звука в {final_output}...")

    command = [
        'ffmpeg',
        '-y',
        '-i', watermarked_video_silent,
        '-i', original_video,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        final_output
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[+] Успешно! Файл со звуком готов: {final_output}")
    except subprocess.CalledProcessError:
        print("[-] Ошибка FFmpeg. Убедитесь, что FFmpeg установлен и доступен в системе.")
    except FileNotFoundError:
        print("[-] FFmpeg не найден. Установите его или добавьте в PATH.")


def embed_watermark_with_audio(video_path, output_path, user_id, strength=20, frames_limit=60):
    temp_silent_output = "temp_silent_" + output_path

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_silent_output, fourcc, fps, (width, height))

    bits = text_to_bits(user_id)
    bits_len = len(bits)

    print(f"[*] Старт маркировки. ID: {user_id}")

    current_frame_idx = 0
    mark_all = total_frames < (frames_limit * 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        is_beginning = current_frame_idx < frames_limit
        is_ending = current_frame_idx >= (total_frames - frames_limit)

        if mark_all or is_beginning or is_ending:
            # DCT
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

        else:
            out.write(frame)

        current_frame_idx += 1
        if current_frame_idx % 100 == 0:
            print(f"Обработано кадров: {current_frame_idx}/{total_frames}", end='\r')

    cap.release()
    out.release()
    print("\n[*] Видеоряд сформирован.")

    add_audio_to_video(video_path, temp_silent_output, output_path)

    if os.path.exists(temp_silent_output):
        os.remove(temp_silent_output)
        print("[*] Временные файлы удалены.")

embed_watermark_with_audio('input1.mp4', 'MBEAST.mp4', 22225555)