import os
import sounddevice as sd
import soundfile as sf
import time
import numpy as np

# 저장 경로 (하위 폴더 없이)
negative_path = "C:\\AI\\multiclass_dataset\\negative"
os.makedirs(negative_path, exist_ok=True)

def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data)) * noise_level
    return data + noise.reshape(-1, 1)

def change_volume(data, factor=1.5):
    return np.clip(data * factor, -1.0, 1.0)

filename = input("enter the name:")

for i in range(5):
    print(f"\n{filename}_{i+1} 녹음을 시작합니다.\n")
    duration = 5

    filename_base = f"{filename}_{i+1}"
    filepath = os.path.join(negative_path, f"{filename_base}.wav")

    # 녹음
    print(f"\n녹음을 시작합니다. 5초 동안 아무 말도 하지 않거나, 다른 사람이 말하게 해보세요...")
    recording = sd.rec(int(16000 * duration), samplerate=16000, channels=1)
    sd.wait()

    sf.write(filepath, recording, 16000)
    print(f"\n녹음 완료! 저장 위치: {filepath}")

    # === 데이터 증강 === (같은 폴더에 저장)
    # 노이즈 추가
    noisy = add_noise(recording)
    noisy_path = os.path.join(negative_path, f"{filename_base}_noisy.wav")
    sf.write(noisy_path, noisy, 16000)

    # 볼륨 조정
    louder = change_volume(recording, factor=1.5)
    louder_path = os.path.join(negative_path, f"{filename_base}_louder.wav")
    sf.write(louder_path, louder, 16000)

    print(f"증강된 데이터 저장 완료: \n - {noisy_path}\n - {louder_path}")

    time.sleep(1)
