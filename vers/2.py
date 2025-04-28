# -*- coding: utf-8 -*-
# ==============================================================================
#         Streamlit Приложение для rPPG Измерения Пульса в Реальном Времени
#      (Все исследованные методы) v12.1 - Полный Код Скользящего Окна
# ==============================================================================
import streamlit as st
import cv2
import numpy as np
import time
import collections
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks, butter, filtfilt, detrend
from sklearn.decomposition import FastICA
import traceback
import os
import warnings

# --- Отключение предупреждений ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Конфигурация Приложения ---
APP_TITLE = "rPPG Детектор Пульса (Все Методы + Скользящее Окно НС)"
WINDOW_SIZE_SEC = 10; FPS_ESTIMATED = 30; BUFFER_SIZE = WINDOW_SIZE_SEC * FPS_ESTIMATED
ROI_PADDING_FACTOR = 0.1; HR_WINDOW_SEC_ANALYSIS = 6; HR_MIN = 40; HR_MAX = 180
BANDPASS_LOW = 0.7; BANDPASS_HIGH = 4.0; MIN_DETECTION_CONFIDENCE = 0.6
HR_SMOOTHING_FACTOR = 0.3 # Фактор сглаживания ЧСС (0.0 - без сглаживания, ~0.1-0.3 - умеренное)

# --- Настройки Нейросетей ---
NEURAL_NETWORKS = ['SimplePhysNet', 'ImprovedPhysNet', 'HAFNet', 'PhysFormer']
NN_MODEL_PATHS = { # <<< УКАЖИТЕ ПРАВИЛЬНЫЕ ПУТИ К ВАШИМ .pth ФАЙЛАМ! >>>
    'SimplePhysNet': 'SimplePhysNet_final.pth', # Пример: '/path/to/your/models/SimplePhysNet_final.pth'
    'ImprovedPhysNet': 'ImprovedPhysNet_final.pth',
    'HAFNet': 'HAFNet_final.pth',
    'PhysFormer': 'PhysFormer_final.pth',
}
NN_WINDOW_SIZE_FRAMES = 150  # Размер окна, на котором ОБУЧЕНЫ модели (НЕ МЕНЯТЬ ЛЕГКОМЫСЛЕННО!)
# --- Шаг скользящего окна НС ---
# Предсказание будет выполняться каждые N кадров ПОСЛЕ заполнения буфера.
NN_SLIDE_STEP = 30           # Значение по умолчанию
NN_RESIZE_DIM = (64, 64)
PHYSNET_DROPOUT = 0.3
HAFNET_FEATURE_DIM = 32; HAFNET_TRANSFORMER_LAYERS = 1; HAFNET_TRANSFORMER_HEADS = 4; HAFNET_DROPOUT = 0.15
PHYSFORMER_FEATURE_DIM = 64; PHYSFORMER_TRANSFORMER_LAYERS = 2; PHYSFORMER_TRANSFORMER_HEADS = 4; PHYSFORMER_DROPOUT = 0.1

# --- Настройка Устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# --- БЛОК: Вспомогательные Функции (Нормализация, Фильтрация, ЧСС) ---
def normalize_signal_np(signal):
    if signal is None or signal.size == 0: return np.array([])
    mean_val = np.mean(signal); std_val = np.std(signal)
    if std_val < 1e-8: return signal - mean_val
    return (signal - mean_val) / std_val

def bandpass_filter(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    if signal is None or signal.size < order * 3 + 1 or fs <= 0: return signal
    nyq = 0.5 * fs; low_f = max(0.01, low); high_f = min(nyq - 0.01, high)
    if low_f >= high_f: return signal
    low_norm = low_f / nyq; high_norm = high_f / nyq
    try:
        b, a = butter(order, [low_norm, high_norm], btype='band')
        y = filtfilt(b, a, signal, method="gust")
    except ValueError as e:
        # print(f"Ошибка фильтрации: {e}, длина сигнала: {len(signal)}, fs: {fs}")
        return signal
    if np.isnan(y).any(): return signal
    return y

def calculate_hr(signal, fs, window_sec=HR_WINDOW_SEC_ANALYSIS, min_hr=HR_MIN, max_hr=HR_MAX):
    if signal is None or fs <= 0: return np.nan
    effective_window_samples = min(len(signal), int(window_sec * fs))
    if effective_window_samples < fs * 1.0: # Требуем хотя бы 1 секунду данных
         return np.nan

    segment = signal[-effective_window_samples:]
    min_dist = max(1, int(fs / (max_hr / 60.0))) if max_hr > 0 else 1

    hr = np.nan
    segment_std = np.std(segment)
    if not np.isnan(segment).any() and segment_std > 1e-6:
        try:
            segment_median = np.median(segment)
            peaks, _ = find_peaks(segment, distance=min_dist, height=segment_median)

            if len(peaks) > 1: # Нужно хотя бы 2 пика
                actual_segment_duration_sec = len(segment) / fs
                peaks_per_sec = len(peaks) / actual_segment_duration_sec if actual_segment_duration_sec > 0 else 0
                hr_calc = peaks_per_sec * 60.0
                if min_hr <= hr_calc <= max_hr:
                    hr = hr_calc
        except Exception as e:
            # print(f"Ошибка расчета ЧСС: {e}")
            pass
    return hr

# --- БЛОК: Классические Методы (POS, CHROM, ICA) ---
def chrom_method(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0]<2 or rgb_buffer.shape[1]!=3: return np.array([])
    std_rgb=np.std(rgb_buffer,axis=0,keepdims=True); std_rgb[std_rgb<1e-8]=1.0; RGB_std=rgb_buffer/std_rgb
    X=3*RGB_std[:,0]-2*RGB_std[:,1]; Y=1.5*RGB_std[:,0]+RGB_std[:,1]-1.5*RGB_std[:,2]
    std_X=np.std(X); std_Y=np.std(Y); alpha=std_X/(std_Y+1e-8) if std_Y>1e-8 else 1.0
    chrom_signal=X-alpha*Y;
    try: return detrend(chrom_signal)
    except ValueError: return chrom_signal - np.mean(chrom_signal)

def pos_method(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0]<2 or rgb_buffer.shape[1]!=3: return np.array([])
    mean_rgb=np.mean(rgb_buffer,axis=0,keepdims=True); mean_rgb[mean_rgb<1e-8]=1.0; rgb_norm=rgb_buffer/mean_rgb
    try: rgb_detrended=detrend(rgb_norm,axis=0)
    except ValueError: rgb_detrended=rgb_norm-np.mean(rgb_norm,axis=0,keepdims=True)
    proj_mat=np.array([[0,1,-1],[-2,1,1]]); proj_sig=np.dot(rgb_detrended,proj_mat.T); std_dev=np.std(proj_sig,axis=0)
    if std_dev.shape[0]<2 or std_dev[0]<1e-8: alpha=1.0
    else: alpha=std_dev[1]/(std_dev[0]+1e-8)
    pos_signal=proj_sig[:,0]+alpha*proj_sig[:,1];
    try: return detrend(pos_signal)
    except ValueError: return pos_signal - np.mean(pos_signal)

def ica_method(rgb_buffer, fs_approx=30):
    if rgb_buffer is None or rgb_buffer.shape[0] < 3 or rgb_buffer.shape[1] != 3: return np.array([])
    try: rgb_detrended=detrend(rgb_buffer,axis=0)
    except ValueError: rgb_detrended=rgb_buffer-np.mean(rgb_buffer,axis=0,keepdims=True)
    n_components=3
    try: # Проверка ранга
        cov_matrix = np.cov(rgb_detrended.T)
        rank = np.linalg.matrix_rank(cov_matrix)
        n_components = max(1, rank)
    except Exception: pass # Используем n_components=3 по умолчанию

    if n_components == 1: # Если ранг 1, ICA не нужен
        stds = np.std(rgb_detrended, axis=0)
        if len(stds) > 0 and np.max(stds) > 1e-8:
            best_channel = rgb_detrended[:, np.argmax(stds)]
            try: return detrend(best_channel)
            except ValueError: return best_channel - np.mean(best_channel)
        else: return np.zeros(rgb_buffer.shape[0])

    S_ = np.array([])
    try:
        ica = FastICA(n_components=n_components, random_state=42, max_iter=250, tol=0.08, whiten='unit-variance')
        S_ = ica.fit_transform(rgb_detrended)
        if S_.shape[1] != n_components:
             # print(f"Warning ICA: Несоответствие размерности выхода ICA ({S_.shape[1]} != {n_components})")
             return np.array([])
    except Exception as e:
        # print(f"Warning ICA fit_transform: {e}")
        return np.array([])

    best_idx=-1; max_power=-1; low_f=BANDPASS_LOW; high_f=BANDPASS_HIGH
    for i in range(S_.shape[1]):
        sig=S_[:,i]
        if len(sig)<2 or np.std(sig)<1e-8: continue
        try:
            fft_win=sig*np.hanning(len(sig)); fft_v=np.fft.rfft(fft_win)
            fft_f=np.fft.rfftfreq(len(sig),1.0/fs_approx if fs_approx > 0 else 1.0/FPS_ESTIMATED)
            p_s=np.abs(fft_v)**2; mask=(fft_f>=low_f)&(fft_f<=high_f)
            if np.any(mask):
                p_band=np.mean(p_s[mask])
                if p_band > max_power: max_power=p_band; best_idx=i
        except Exception: continue

    if best_idx == -1: # Если не нашли по FFT, выбираем по std
        stds = np.std(S_,axis=0)
        if len(stds) > 0 and np.max(stds) > 1e-8: best_idx = np.argmax(stds)
        else: return np.array([]) # Не удалось выбрать

    selected_component = S_[:, best_idx]
    try: return detrend(selected_component)
    except ValueError: return selected_component - np.mean(selected_component)


# --- БЛОК: Определение Моделей Нейронных Сетей ---
# (ВКЛЮЧЕНЫ ВСЕ КЛАССЫ)
class SimplePhysNet(nn.Module):
    def __init__(self,in_channels=3,out_len=NN_WINDOW_SIZE_FRAMES): super().__init__(); self.encoder=nn.Sequential(nn.Conv3d(in_channels,32,(1,5,5),padding=(0,2,2)),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(32,64,(3,3,3),padding=1),nn.BatchNorm3d(64),nn.ReLU(),nn.MaxPool3d((1,2,2))); self.decoder=nn.Sequential(nn.Conv3d(64,1,kernel_size=1),nn.AdaptiveAvgPool3d((out_len,1,1)))
    def forward(self,x): x=self.encoder(x); x=self.decoder(x); x=x.squeeze(-1).squeeze(-1).squeeze(1); return x
class ImprovedPhysNet(nn.Module):
    def __init__(self,in_channels=3,out_len=NN_WINDOW_SIZE_FRAMES,dropout=PHYSNET_DROPOUT): super().__init__(); self.encoder=nn.Sequential(nn.Conv3d(in_channels,32,(1,5,5),padding=(0,2,2)),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(32,64,(3,3,3),padding=1),nn.BatchNorm3d(64),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(64,128,(3,3,3),padding=1),nn.BatchNorm3d(128),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Dropout3d(dropout)); self.decoder=nn.Sequential(nn.Conv3d(128,1,kernel_size=1),nn.AdaptiveAvgPool3d((out_len,1,1)))
    def forward(self,x): x=self.encoder(x); x=self.decoder(x); x=x.squeeze(-1).squeeze(-1).squeeze(1); return x
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=512): super().__init__(); pos=torch.arange(max_len).unsqueeze(1); div=torch.exp(torch.arange(0,d_model,2)*(-np.log(10000.0)/d_model)); pe=torch.zeros(max_len,d_model); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:x.size(1)].unsqueeze(0)
class PhysFormer3DStem(nn.Module):
    def __init__(self,in_channels=3,feature_dim=64): super().__init__(); self.feature_dim_arg = feature_dim; self.stem=nn.Sequential(nn.Conv3d(in_channels,16,(1,5,5),padding=(0,2,2)),nn.BatchNorm3d(16),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(16,32,(3,3,3),padding=1),nn.BatchNorm3d(32),nn.ReLU(),nn.MaxPool3d((1,2,2)),nn.Conv3d(32,feature_dim,(3,3,3),padding=1),nn.BatchNorm3d(feature_dim),nn.ReLU(),nn.AdaptiveAvgPool3d((None,1,1)))
    def forward(self,x): x=self.stem(x); x=x.squeeze(-1).squeeze(-1); x=x.permute(0,2,1); return x
class TemporalDifferenceModule(nn.Module):
    def __init__(self): super().__init__()
    def forward(self,x): diff=x[:,1:,:]-x[:,:-1,:]; diff=F.pad(diff,(0,0,1,0),"constant",0); return diff
class PhysFormer(nn.Module):
    def __init__(self,in_channels=3,feature_dim=PHYSFORMER_FEATURE_DIM,num_layers=PHYSFORMER_TRANSFORMER_LAYERS,num_heads=PHYSFORMER_TRANSFORMER_HEADS,dropout=PHYSFORMER_DROPOUT,max_len=NN_WINDOW_SIZE_FRAMES):
        super().__init__(); self.feat_ext=PhysFormer3DStem(in_channels,feature_dim); self.tdm=TemporalDifferenceModule(); self.pos_enc=PositionalEncoding(feature_dim,max_len)
        enc_l=nn.TransformerEncoderLayer(d_model=feature_dim,nhead=num_heads,dim_feedforward=feature_dim*2,dropout=dropout,activation='relu',batch_first=True)
        self.transformer=nn.TransformerEncoder(enc_l,num_layers=num_layers,norm=nn.LayerNorm(feature_dim)); self.head=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,1))
    def forward(self,x): feat=self.feat_ext(x); diff=self.tdm(feat); diff_pos=self.pos_enc(diff); z=self.transformer(diff_pos); out=self.head(z); return out.squeeze(-1)
class HAFNet(nn.Module):
    def __init__(self,in_channels=3,feature_dim=HAFNET_FEATURE_DIM,nhead=HAFNET_TRANSFORMER_HEADS,num_encoder_layers=HAFNET_TRANSFORMER_LAYERS,dropout=HAFNET_DROPOUT,max_len=NN_WINDOW_SIZE_FRAMES):
        super().__init__(); self.cnn_stem=PhysFormer3DStem(in_channels,feature_dim); self.pos_enc=PositionalEncoding(feature_dim,max_len)
        enc_l=nn.TransformerEncoderLayer(d_model=feature_dim,nhead=nhead,dim_feedforward=feature_dim*2,dropout=dropout,activation='relu',batch_first=True)
        self.transformer_enc=nn.TransformerEncoder(enc_l,num_layers=num_encoder_layers,norm=nn.LayerNorm(feature_dim)); self.head=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,1))
    def forward(self,x): feat=self.cnn_stem(x); feat_pos=self.pos_enc(feat); mem=self.transformer_enc(feat_pos); out=self.head(mem); return out.squeeze(-1)

# --- Функция Загрузки Моделей НС ---
def get_model_instance(model_name, model_path):
    """Создает экземпляр модели и загружает веса."""
    model_class_map = {
        "SimplePhysNet": SimplePhysNet, "ImprovedPhysNet": ImprovedPhysNet,
        "HAFNet": HAFNet, "PhysFormer": PhysFormer
    }
    model_class = model_class_map.get(model_name)
    if not model_class:
        raise ValueError(f"Неизвестный класс модели для {model_name}")

    model = model_class().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Обработка префикса 'module.'
        if all(k.startswith('module.') for k in state_dict.keys()):
             state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        model_dict = model.state_dict()
        # Загружаем только совпадающие слои
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        if not state_dict:
             print(f"  Предупреждение: Не найдено совпадающих слоев в state_dict для {model_name} из {model_path}")
             return None

        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False) # strict=False полезно
        model.eval()
        print(f"  Успешно загружена модель: {model_name}")
        return model
    except FileNotFoundError:
        print(f"  Ошибка: Файл не найден - {model_path}")
        return None
    except Exception as e:
        print(f"  Ошибка загрузки весов для {model_name} из {model_path}: {e}")
        traceback.print_exc()
        return None


# --- Функция Загрузки Ресурсов с Кешированием ---
@st.cache_resource
def load_resources():
    models = {}
    print("--- Загрузка моделей НС ---")
    for name in NEURAL_NETWORKS:
        path = NN_MODEL_PATHS.get(name)
        if not path:
            print(f"  Предупреждение: Путь не указан для модели {name}, пропускаем.")
            continue

        model_file_exists = os.path.exists(path)
        print(f"Проверка файла '{path}' для модели {name}: {'Найден' if model_file_exists else 'НЕ НАЙДЕН'}")

        if model_file_exists:
            model_instance = get_model_instance(name, path)
            if model_instance:
                models[name] = model_instance
        else:
            print(f"  Пропуск загрузки {name}, т.к. файл не найден.")

    print(f"--- Загружено {len(models)} НС моделей ---")

    print("--- Загрузка MediaPipe ---")
    face_detector = None
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(min_detection_confidence=MIN_DETECTION_CONFIDENCE, model_selection=0)
        print("MediaPipe Face Detection успешно загружен.")
    except Exception as e:
        print(f"Критическая ошибка инициализации MediaPipe: {e}")
        traceback.print_exc()
        st.error(f"Не удалось загрузить MediaPipe Face Detection: {e}. Приложение не может продолжить.")
        st.stop() # Останавливаем, если MP не работает

    return models, face_detector

# --- Основная Функция Приложения ---
def run_rppg_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # --- Загрузка Ресурсов ---
    loaded_nn_models, face_detector = load_resources()
    available_methods = ["POS", "CHROM", "ICA"] + sorted(list(loaded_nn_models.keys()))
    if not available_methods: st.error("Ошибка: Не доступно ни одного метода rPPG."); return

    selected_method = st.sidebar.selectbox("Выберите метод измерения:", available_methods, index=0, key="method_select")
    # Слайдер для шага НС вынесен сюда для инициализации значения
    nn_slide_step_option = st.sidebar.slider(
        "Шаг скольж. окна НС (кадры):",
        min_value=5, max_value=NN_WINDOW_SIZE_FRAMES,
        value=NN_SLIDE_STEP, step=5, key="nn_stride",
        help="Как часто выполнять предсказание НС после заполнения буфера. Меньше = чаще, но выше нагрузка.",
        disabled=(selected_method not in loaded_nn_models) # Делаем неактивным для классики
    )

    # --- Инициализация Камеры ---
    cap = None
    camera_indices = [0, 1, 2]
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Веб-камера успешно открыта (индекс {idx}).")
                res_w = 640; res_h = 480 # Попытка установить разрешение
                if cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h): print(f"Установлено разрешение камеры: {res_w}x{res_h}")
                else: print("Не удалось установить разрешение камеры.")
                break
            else: cap.release(); cap = None
        except Exception as e: print(f"Ошибка при попытке открыть камеру {idx}: {e}"); cap = None
    if cap is None: st.error("Не удалось открыть веб-камеру."); return

    # --- Инициализация Буферов и Переменных в session_state ---
    if 'initialized' not in st.session_state:
        st.session_state.mean_rgb_buffer = collections.deque(maxlen=BUFFER_SIZE)
        st.session_state.rppg_signal_buffer = collections.deque(maxlen=BUFFER_SIZE) # Для классики
        st.session_state.timestamps = collections.deque(maxlen=BUFFER_SIZE)
        st.session_state.nn_frame_buffer = collections.deque(maxlen=NN_WINDOW_SIZE_FRAMES) # Для НС
        st.session_state.current_hr = np.nan
        st.session_state.last_fps_time = time.time()
        st.session_state.frame_count_for_fps = 0
        st.session_state.last_signal_time = time.time()
        st.session_state.nn_buffer_filled = False
        st.session_state.nn_frames_since_last_pred = 0
        st.session_state.last_method = selected_method
        st.session_state.initialized = True
        print("Состояние сессии инициализировано.")

    # Сброс при смене метода
    if selected_method != st.session_state.last_method:
        print(f"Метод изменен. Сброс буферов и состояния НС.")
        st.session_state.mean_rgb_buffer.clear(); st.session_state.rppg_signal_buffer.clear()
        st.session_state.timestamps.clear(); st.session_state.nn_frame_buffer.clear()
        st.session_state.current_hr = np.nan; st.session_state.nn_buffer_filled = False
        st.session_state.nn_frames_since_last_pred = 0; st.session_state.last_method = selected_method
        # Обновляем состояние слайдера
        st.rerun() # Перезапуск скрипта для обновления активности слайдера

    # --- Интерфейс ---
    col1, col2 = st.columns([3, 1])
    with col1: stframe = st.empty()
    with col2:
        st.subheader("Параметры и Результат")
        st.write(f"Метод: **{selected_method}**")
        if selected_method in loaded_nn_models:
             st.write(f"Шаг НС: {nn_slide_step_option} кадров") # Показываем текущий шаг
        hr_placeholder = st.empty(); hr_placeholder.metric("Текущий Пульс (уд/мин)", "Ожидание...")
        fps_placeholder = st.empty(); fps_placeholder.write("Камера FPS: ...")
        signal_placeholder = st.empty()
        status_placeholder = st.empty()

    # --- Главный Цикл ---
    is_running = True
    mp_duration = 0.0 # Инициализация для вывода FPS
    print("Запуск основного цикла приложения...")
    while is_running:
        loop_start_time = time.time()
        try:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                status_placeholder.warning("Проблема с чтением кадра...", icon="⚠️"); time.sleep(0.1)
                continue

            current_time = time.time(); st.session_state.timestamps.append(current_time)

            # --- Обработка Кадра (MediaPipe) ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            if frame_rgb.dtype != np.uint8: frame_rgb = frame_rgb.astype(np.uint8)
            mp_start_time = time.time()
            results = face_detector.process(frame_rgb)
            mp_duration = time.time() - mp_start_time # Время выполнения MP
            frame_rgb.flags.writeable = True

            height, width, _ = frame.shape; face_found = False; roi_display = frame.copy()
            should_predict_nn = False # Флаг: нужно ли делать предсказание НС

            if results.detections:
                detection = sorted(results.detections, key=lambda x: x.score[0], reverse=True)[0]
                bboxC = detection.location_data.relative_bounding_box
                if bboxC and detection.score[0] > MIN_DETECTION_CONFIDENCE:
                    xmin=int(bboxC.xmin*width); ymin=int(bboxC.ymin*height); w=int(bboxC.width*width); h=int(bboxC.height*height)
                    pad_w=int(w*ROI_PADDING_FACTOR); pad_h=int(h*ROI_PADDING_FACTOR)
                    x1=max(0,xmin-pad_w); y1=max(0,ymin-pad_h); x2=min(width,xmin+w+pad_w); y2=min(height,ymin+h+pad_h)
                    if y2 > y1 and x2 > x1:
                        face_found = True; roi = frame[y1:y2, x1:x2]
                        # 1. Данные для классических методов
                        mean_bgr_roi = np.mean(roi, axis=(0, 1))
                        st.session_state.mean_rgb_buffer.append(mean_bgr_roi[::-1])
                        st.session_state.last_signal_time = current_time

                        # 2. Данные для НС методов (добавляем всегда, если метод НС выбран)
                        if selected_method in loaded_nn_models:
                            try:
                                # Выполняем предобработку только если выбран НС метод
                                prep_start = time.time()
                                roi_resized_nn = cv2.resize(roi, NN_RESIZE_DIM, interpolation=cv2.INTER_LINEAR)
                                roi_norm = (roi_resized_nn / 255.0) * 2.0 - 1.0
                                roi_norm_transposed = roi_norm.transpose(2, 0, 1) # (C, H, W)
                                st.session_state.nn_frame_buffer.append(roi_norm_transposed) # Добавляем в deque
                                # print(f"NN Prep time: {(time.time() - prep_start)*1000:.1f} ms")

                                # --- Логика Скользящего Окна ---
                                current_buffer_len = len(st.session_state.nn_frame_buffer)

                                if not st.session_state.nn_buffer_filled:
                                    if current_buffer_len == NN_WINDOW_SIZE_FRAMES:
                                        st.session_state.nn_buffer_filled = True
                                        should_predict_nn = True # Первое предсказание
                                        st.session_state.nn_frames_since_last_pred = 0
                                        status_placeholder.success("НС буфер заполнен, первая обработка...", icon="✅")
                                    else:
                                        status_placeholder.info(f"Сбор данных для НС: {current_buffer_len}/{NN_WINDOW_SIZE_FRAMES} кадров", icon="⏳")
                                else: # Буфер уже был заполнен
                                    st.session_state.nn_frames_since_last_pred += 1
                                    if st.session_state.nn_frames_since_last_pred >= nn_slide_step_option:
                                        should_predict_nn = True # Пора делать предсказание
                                        st.session_state.nn_frames_since_last_pred = 0
                                        status_placeholder.success(f"НС обработка (шаг {nn_slide_step_option})...", icon="⚡")
                                    # Можно раскомментировать для показа ожидания
                                    # elif not should_predict_nn:
                                    #     remaining = nn_slide_step_option - st.session_state.nn_frames_since_last_pred
                                    #     status_placeholder.info(f"НС: след. через {remaining} к.", icon="⏳")

                            except cv2.error as e_resize: pass # Ошибка ресайза/обработки

                        cv2.rectangle(roi_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # --- Управление состоянием, если лицо не найдено ---
            if not face_found:
                 if st.session_state.nn_buffer_filled:
                      print("Лицо потеряно, сброс состояния НС буфера.")
                      st.session_state.nn_buffer_filled = False # Сбрасываем флаг заполненности
                 st.session_state.nn_frames_since_last_pred = 0 # Сбрасываем счетчик шага
                 if time.time() - st.session_state.last_signal_time > 1.0:
                      status_placeholder.info("Лицо не обнаружено...", icon="🚫")


            # --- Расчет rPPG Сигнала и ЧСС ---
            calculated_hr_this_step = np.nan; actual_fps = FPS_ESTIMATED
            if len(st.session_state.timestamps) > 10:
                 time_diff = st.session_state.timestamps[-1] - st.session_state.timestamps[-10]
                 if time_diff > 0.01: actual_fps = 9 / time_diff
                 actual_fps = max(1.0, min(actual_fps, 60.0))

            signal_processed_success = False; current_rppg_signal_for_plot = None

            # --- Классические методы ---
            if selected_method in ["POS", "CHROM", "ICA"]:
                 min_len_classical = int(BUFFER_SIZE * 0.3);
                 if selected_method == "ICA": min_len_classical = int(BUFFER_SIZE * 0.7)
                 if len(st.session_state.mean_rgb_buffer) > min_len_classical:
                    rgb_data = np.array(st.session_state.mean_rgb_buffer); rppg_signal_raw = np.array([])
                    try: # Вызов метода
                        if selected_method == "POS": rppg_signal_raw = pos_method(rgb_data)
                        elif selected_method == "CHROM": rppg_signal_raw = chrom_method(rgb_data)
                        elif selected_method == "ICA": rppg_signal_raw = ica_method(rgb_data, actual_fps)
                    except Exception as e_method: print(f"Ошибка метода {selected_method}: {e_method}")
                    if rppg_signal_raw is not None and rppg_signal_raw.size > 1: # Обработка сигнала
                        rppg_norm = normalize_signal_np(rppg_signal_raw); rppg_filt = bandpass_filter(rppg_norm, actual_fps)
                        if rppg_filt is not None and rppg_filt.size > 1: # Обновление буфера rPPG
                            num_already = len(st.session_state.rppg_signal_buffer); num_expected = len(rgb_data); num_new = num_expected - num_already
                            if num_new > 0 and len(rppg_filt) >= num_new: st.session_state.rppg_signal_buffer.extend(rppg_filt[-num_new:])
                            elif len(rppg_filt) > num_already : st.session_state.rppg_signal_buffer.extend(rppg_filt[num_already:])
                            elif len(rppg_filt)>0 and num_already==0: st.session_state.rppg_signal_buffer.extend(rppg_filt)
                            signal_processed_success = True; current_rppg_signal_for_plot = np.array(st.session_state.rppg_signal_buffer)
                            min_len_hr = int(HR_WINDOW_SEC_ANALYSIS * actual_fps * 0.8) # Расчет ЧСС
                            if len(st.session_state.rppg_signal_buffer) > min_len_hr:
                                hr_val = calculate_hr(current_rppg_signal_for_plot, actual_fps, window_sec=HR_WINDOW_SEC_ANALYSIS)
                                if not np.isnan(hr_val): calculated_hr_this_step = hr_val

            # --- Нейросетевые методы ---
            elif selected_method in loaded_nn_models:
                 if should_predict_nn: # Делаем предсказание?
                    rppg_signal_raw_nn = np.array([]); nn_predict_start_time = time.time()
                    try:
                        if len(st.session_state.nn_frame_buffer) == NN_WINDOW_SIZE_FRAMES: # Проверка полноты буфера
                            model_instance = loaded_nn_models[selected_method]
                            nn_input_numpy = np.array(st.session_state.nn_frame_buffer) # (T, C, H, W)
                            nn_input_permuted = nn_input_numpy.transpose(1, 0, 2, 3)     # (C, T, H, W)
                            input_tensor = torch.from_numpy(nn_input_permuted).float().unsqueeze(0).to(device) # (B=1, C, T, H, W)

                            with torch.no_grad(): rppg_signal_raw_nn = model_instance(input_tensor).squeeze().cpu().numpy()
                            nn_predict_duration = time.time() - nn_predict_start_time
                            print(f"NN ({selected_method}) prediction time: {nn_predict_duration:.3f}s")
                        else: print(f"Предупреждение: Попытка предсказания НС, но буфер не полон.")

                    except Exception as e_method: print(f"Ошибка предсказания НС {selected_method}: {e_method}"); traceback.print_exc()

                    if rppg_signal_raw_nn is not None and rppg_signal_raw_nn.size > 1: # Обработка выхода НС
                        rppg_norm_nn = normalize_signal_np(rppg_signal_raw_nn); rppg_filt_nn = bandpass_filter(rppg_norm_nn, actual_fps)
                        if rppg_filt_nn is not None and rppg_filt_nn.size > 1:
                            signal_processed_success = True; current_rppg_signal_for_plot = rppg_filt_nn
                            nn_window_sec = NN_WINDOW_SIZE_FRAMES / actual_fps if actual_fps > 5 else (NN_WINDOW_SIZE_FRAMES / FPS_ESTIMATED)
                            min_samples_hr = int(actual_fps * 1.5) # Расчет ЧСС из выхода НС
                            if len(rppg_filt_nn) >= min_samples_hr:
                                hr_val = calculate_hr(rppg_filt_nn, actual_fps, window_sec=nn_window_sec)
                                if not np.isnan(hr_val): calculated_hr_this_step = hr_val
                 # Если не делали предсказание на этом шаге, но сигнал уже был обработан ранее (для графика)
                 elif signal_processed_success is False and current_rppg_signal_for_plot is None and len(st.session_state.rppg_signal_buffer)>0 and st.session_state.last_method==selected_method :
                      # Можно отобразить предыдущий сигнал НС, если он есть и метод не менялся
                      # current_rppg_signal_for_plot = np.array(st.session_state.rppg_signal_buffer) # Пример, если бы мы сохраняли его
                      pass # Пока просто не обновляем график


            # --- Сглаживание и Обновление состояния ЧСС ---
            smoothed_hr = st.session_state.current_hr
            if not np.isnan(calculated_hr_this_step):
                 if np.isnan(st.session_state.current_hr): smoothed_hr = calculated_hr_this_step
                 else: smoothed_hr = (HR_SMOOTHING_FACTOR * calculated_hr_this_step + (1 - HR_SMOOTHING_FACTOR) * st.session_state.current_hr)
                 st.session_state.current_hr = smoothed_hr # Обновляем сохраненное

            # --- Обновление Интерфейса ---
            hr_display = f"{st.session_state.current_hr:.1f}" if not np.isnan(st.session_state.current_hr) else "..."
            hr_placeholder.metric("Текущий Пульс (уд/мин)", hr_display)
            st.session_state.frame_count_for_fps += 1; elapsed_time = time.time() - st.session_state.last_fps_time
            if elapsed_time > 1.0: # Обновляем FPS раз в секунду
                display_fps = st.session_state.frame_count_for_fps / elapsed_time
                fps_placeholder.write(f"Камера FPS: {display_fps:.1f} | MP time: {mp_duration*1000:.1f} ms | Обр. FPS: {actual_fps:.1f}")
                st.session_state.last_fps_time = time.time(); st.session_state.frame_count_for_fps = 0
            if signal_processed_success and current_rppg_signal_for_plot is not None and current_rppg_signal_for_plot.size > 10:
                 plot_data = current_rppg_signal_for_plot[-BUFFER_SIZE:] # Показываем хвост
                 signal_placeholder.line_chart(plot_data)
            stframe.image(roi_display, channels="BGR", use_container_width=True)

        except KeyboardInterrupt:
            is_running = False
            print("\nПрерывание пользователем (Ctrl+C)")
            st.warning("Приложение остановлено пользователем", icon="⚠️")
        except Exception as e_main:
            print(f"\nКритическая ошибка в главном цикле: {e_main}")
            traceback.print_exc()
            is_running = False
            try:
                error_msg = f"Произошла критическая ошибка:\n{str(e_main)}\n\nПопробуйте перезапустить приложение."
                st.error(error_msg, icon="🚨")
                status_placeholder.error("Остановка приложения...", icon="⛔")
            except Exception as e_error:
                print(f"Дополнительная ошибка при отображении сообщения: {e_error}")
            time.sleep(2)

    # --- Освобождение Ресурсов ---
    print("\n=== Завершение работы приложения ===")
    try:
        if face_detector:
            face_detector.close()
            print("MediaPipe Face Detection освобожден")
    except Exception as e:
        print(f"Ошибка при освобождении MediaPipe: {e}")

    try:
        if cap is not None:
            if cap.isOpened():
                cap.release()
                print("Веб-камера успешно освобождена")
            else:
                print("Веб-камера уже была закрыта")
    except Exception as e:
        print(f"Ошибка при освобождении камеры: {e}")

    # Очистка CUDA кеша, если использовался
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            print("CUDA кеш очищен")
        except Exception as e:
            print(f"Ошибка при очистке CUDA: {e}")

    print("Приложение успешно завершено\n")

# --- Запуск Приложения ---
if __name__ == "__main__":
    print("\n=== Инициализация rPPG приложения ===")
    
    # Проверка наличия моделей
    missing_models = []
    for name, path in NN_MODEL_PATHS.items():
        if not path:
            print(f"⚠️ Путь не указан для модели {name}")
            continue
        if not os.path.exists(path):
            missing_models.append(name)
            print(f"❌ Модель не найдена: {name} ({path})")
        else:
            print(f"✓ Модель доступна: {name}")
    
    if missing_models and NEURAL_NETWORKS:
        print("\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Следующие модели НС недоступны:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nКлассические методы (POS, CHROM, ICA) останутся доступны.")
        print("Для использования НС методов укажите правильные пути в NN_MODEL_PATHS.")
    
    print("\nЗапуск приложения...")
    run_rppg_app()