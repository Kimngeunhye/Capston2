import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pyaudio
import wave
import noisereduce as nr
from keras.layers import LSTM, Bidirectional, Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from keras import models
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from keras.layers import LSTM, Bidirectional, Dense, Input, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras import models

def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Conv1D Layers
    model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Bidirectional LSTM Layer
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))

    # Global Average Pooling
    model.add(GlobalAveragePooling1D())

    # Dense Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(f"{num_classes}개 클래스를 가진 다중 분류 모델로 학습합니다.")
    return model



def extract_mfcc(file_path, n_mfcc=13, fixed_length=200):
    # 1. 음성 파일 로드 및 전처리
    y, sr = sf.read(file_path)
    y, _ = librosa.effects.trim(y)

    # 2. MFCC, 델타, 델타-델타 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 3. 스택 (총 39차원)
    combined = np.vstack([mfcc, delta, delta2])  # (39, time)

    # 4. ★ 정규화 (feature마다 평균 0, 표준편차 1로)
    mean = np.mean(combined, axis=1, keepdims=True)
    std = np.std(combined, axis=1, keepdims=True) + 1e-6  # 0으로 나누는 걸 방지
    combined = (combined - mean) / std

    # 5. 고정 길이 맞추기 (패딩 또는 자르기)
    if combined.shape[1] < fixed_length:
        pad_width = fixed_length - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :fixed_length]

    # 6. 시간 축이 앞에 오도록 전치
    return combined.T  # shape: (time, 39)

# 음성 녹음 함수 및 노이즈 제거
def record_audio(filename, duration=5):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    p = pyaudio.PyAudio()
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    frames_per_buffer = 1024

    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
    print("Recording...")
    frames = [stream.read(frames_per_buffer) for _ in range(0, int(rate / frames_per_buffer * duration))]
    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    y, sr = librosa.load(filename, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(filename, reduced_noise, sr)
    print(f"Noise reduction applied to {filename}.")

# 데이터 증강 함수
def augment_audio(y, sr):
    # 피치 변경 (Pitch Shift)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)  # 음성을 2단계 올리기

    # 속도 변경 (Time Stretch)
    y_stretch = librosa.effects.time_stretch(y, rate=1.2)  # 속도 20% 빠르게

    # 노이즈 추가
    noise = np.random.randn(len(y)) * 0.01  # 작은 노이즈 추가
    y_noise = y + noise

    return [y_pitch, y_stretch, y_noise]

# 사용자 등록 함수
def register_user(username):
    folder = f"C:\\AI\\multiclass_dataset\\{username}"

    if os.path.exists(folder):
        print(f"{username}님은 이미 등록된 사용자입니다. 다른 이름을 사용해 주세요.")
        return False

    os.makedirs(folder, exist_ok=True)

    for i in range(5):
        file_path = os.path.join(folder, f"{username}_{i + 1}.wav")
        print(f"{i + 1}번째 음성을 녹음합니다...")
        record_audio(file_path, duration=5)

    print(f"{username}님의 음성이 저장되었습니다.")

    augmented_files = []
    for i in range(5):
        file_path = os.path.join(folder, f"{username}_{i + 1}.wav")
        y, sr = librosa.load(file_path, sr=None)
        augmented_data = augment_audio(y, sr)
        for j, aug_y in enumerate(augmented_data):
            augmented_file = os.path.join(folder, f"{username}_{i + 1}_aug_{j + 1}.wav")
            sf.write(augmented_file, aug_y, sr)
            augmented_files.append(augmented_file)

    print(f"데이터 증강 완료. 총 {len(augmented_files)}개의 샘플이 준비되었습니다.")

    # 회원가입 후 자동으로 모델 재학습
    train_model()
    return True

# 무음 구간을 감지하는 함수
def is_silent(audio_path, silence_threshold=0.01):
    # librosa를 사용하여 오디오 파일을 로드
    y, sr = librosa.load(audio_path, sr=16000)

    # librosa의 trim을 이용해 무음 구간 제거
    trimmed_audio, _ = librosa.effects.trim(y, top_db=20)  # top_db는 무음의 강도를 정의

    # trim된 후에도 음성이 없으면 무음으로 판단
    if len(trimmed_audio) == 0:
        return True  # 무음이라 판단

    return False  # 무음이 아니면 False 반환

def login():
    temp_path = "C:\\AI\\temp\\login.wav"
    record_audio(temp_path, duration=10)

    if is_silent(temp_path):
        print("로그인 실패: 입력된 음성이 무음입니다.")
        return

    model = tf.keras.models.load_model("C:\\AI\\models\\multiclass_voice_model.keras")
    classes = np.load("C:\\AI\\models\\label_classes.npy", allow_pickle=True)

    features = extract_mfcc(temp_path, fixed_length=200).reshape(1, 200, 39)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]
    confidence = prediction[0][predicted_index]

    if confidence < 0.5:
        print("로그인 실패: 신뢰도가 낮아 인증할 수 없습니다.")
        print(f"예측된 사용자: {predicted_label}, 신뢰도: {confidence:.4f}")
        return

    # 'Unknown' 클래스를 등록되지 않은 사용자로 처리
    if predicted_label.lower() == 'unknown':
        print("등록되지 않은 사용자입니다. 로그인 실패.")
    else:
        print(f"로그인한 사용자: {predicted_label}, 환영합니다")
        print(f"예측된 확률: {confidence:.4f}")

def train_model():
    dataset_path = "C:\\AI\\multiclass_dataset"

    if len(os.listdir(dataset_path)) == 0:
        print("등록된 사용자가 없습니다. 최소 1명이 회원가입 후 학습을 시작합니다.")
        return

    X, y = [], []
    for user_folder in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_folder)

        if os.path.isdir(user_path):
            label = 'Unknown' if user_folder.lower() == 'negative' else user_folder

            for file in os.listdir(user_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(user_path, file)
                    features = extract_mfcc(file_path)
                    X.append(features)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    # 레이블 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_classes = le.classes_
    num_classes = len(label_classes)

    # 학습/검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    input_shape = (X.shape[1], X.shape[2])

    # 모델 생성
    model = create_model(input_shape=input_shape, num_classes=num_classes)

    # 콜백 설정
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )

    # 모델 저장
    os.makedirs("C:\\AI\\models", exist_ok=True)
    model.save("C:\\AI\\models\\multiclass_voice_model.keras")
    np.save("C:\\AI\\models\\label_classes.npy", label_classes)

    print("\n모델 학습 완료")
    print(f"최종 학습 정확도: {history.history['accuracy'][-1]:.4f}")
    print(f"최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")


# 메인 실행
if __name__ == "__main__":
    while True:
        print("\n메뉴를 선택하세요:")
        print("1. 로그인")
        print("2. 회원가입")
        print("0. 종료")
        choice = input("선택 (1/2/0): ")

        if choice == '1':
            login()
        elif choice == '2':
            while True:
                username = input("회원가입할 사용자 이름을 입력하세요: ")
                # 회원가입 시 이미 등록된 사용자라면 다시 입력받음
                if register_user(username):
                    break  # 회원가입 성공하면 루프 종료
        elif choice == '0':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")