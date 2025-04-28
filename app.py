# -*- coding: utf-8 -*-
# ==============================================================================
#         Streamlit Приложение для rPPG Измерения Пульса в Реальном Времени
#               (Все исследованные методы) v12.1 - Streamlit WebRTC Адаптация
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
# --- Streamlit WebRTC ---
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode # Removed ClientSettings
import av # Для работы с фреймами в streamlit-webrtc
import threading # Для блокировок доступа к общим ресурсам

# --- Отключение предупреждений ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Игнорируем специфичные предупреждения MediaPipe о Protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# --- Конфигурация Приложения ---
APP_TITLE = "rPPG Детектор Пульса (Streamlit WebRTC)"
WINDOW_SIZE_SEC = 10; FPS_ESTIMATED = 30; BUFFER_SIZE = WINDOW_SIZE_SEC * FPS_ESTIMATED
ROI_PADDING_FACTOR = 0.1; HR_WINDOW_SEC_ANALYSIS = 6; HR_MIN = 40; HR_MAX = 180
BANDPASS_LOW = 0.7; BANDPASS_HIGH = 4.0; MIN_DETECTION_CONFIDENCE = 0.6
# --- Фактор сглаживания ЧСС ---
HR_SMOOTHING_FACTOR = 0.3 # (0.0 - без сглаживания, ~0.1-0.3 - умеренное, >0.5 - сильное)

# --- Настройки Нейросетей ---
NEURAL_NETWORKS = ['SimplePhysNet', 'ImprovedPhysNet', 'HAFNet', 'PhysFormer']
NN_MODEL_PATHS = { # <<< УКАЖИТЕ ПРАВИЛЬНЫЕ ПУТИ К ВАШИМ .pth ФАЙЛАМ! >>>
    # Убедитесь, что эти файлы будут доступны в среде развертывания Streamlit
    # Например, поместите их в тот же каталог или подкаталог и используйте относительные пути
    # Если файл не найден, метод будет недоступен.
    'SimplePhysNet': 'SimplePhysNet_final.pth',
    'ImprovedPhysNet': 'ImprovedPhysNet_final.pth',
    'HAFNet': 'HAFNet_final.pth',
    'PhysFormer': 'PhysFormer_final.pth',
}
NN_WINDOW_SIZE_FRAMES = 90; NN_RESIZE_DIM = (64, 64)
PHYSNET_DROPOUT = 0.3
HAFNET_FEATURE_DIM = 32; HAFNET_TRANSFORMER_LAYERS = 1; HAFNET_TRANSFORMER_HEADS = 4; HAFNET_DROPOUT = 0.15
PHYSFORMER_FEATURE_DIM = 64; PHYSFORMER_TRANSFORMER_LAYERS = 2; PHYSFORMER_TRANSFORMER_HEADS = 4; PHYSFORMER_DROPOUT = 0.1

# --- Настройка Устройства ---
# На сервере Streamlit обычно нет GPU, поэтому принудительно CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Используемое устройство: {device}")

# --- БЛОК: Вспомогательные Функции (Нормализация, Фильтрация, ЧСС) ---
def normalize_signal_np(signal):
    if signal is None or signal.size == 0: return np.array([])
    mean_val = np.mean(signal); std_val = np.std(signal)
    if std_val < 1e-8: return signal - mean_val # Возвращаем центрированный сигнал, если std близко к нулю
    return (signal - mean_val) / std_val

def bandpass_filter(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    if signal is None or signal.size < order * 3 + 1 or fs <= 0: return signal
    nyq = 0.5 * fs; low_f = max(0.01, low); high_f = min(nyq - 0.01, high)
    if low_f >= high_f: return signal # Некорректные границы
    low_norm = low_f / nyq; high_norm = high_f / nyq
    try:
        b, a = butter(order, [low_norm, high_norm], btype='band')
        # Используем padlen для стабильности с короткими сигналами
        padlen = min(len(signal) - 1, max(15, 3 * max(len(b), len(a))))
        y = filtfilt(b, a, signal, method="gust", padlen=padlen) # 'gust' стабильнее для реального времени
    except ValueError as e:
        # print(f"Ошибка фильтрации: {e}, длина сигнала: {len(signal)}, fs: {fs}")
        # Пробуем 'pad' как запасной вариант
        try:
            y = filtfilt(b, a, signal, method="pad", padlen=padlen)
        except ValueError:
            return signal # Если и pad не сработал
    if np.isnan(y).any(): return signal
    return y

def calculate_hr(signal, fs, window_sec=HR_WINDOW_SEC_ANALYSIS, min_hr=HR_MIN, max_hr=HR_MAX):
    if signal is None or fs <= 0: return np.nan
    effective_window_samples = min(len(signal), int(window_sec * fs))
    if effective_window_samples < fs * 1.0: return np.nan # Требуем хотя бы 1 секунду данных

    segment = signal[-effective_window_samples:]
    min_dist = int(fs / (max_hr / 60.0)) if max_hr > 0 else 1
    min_dist = max(1, min_dist) # Минимальное расстояние между пиками должно быть хотя бы 1

    hr = np.nan
    segment_std = np.std(segment)
    if not np.isnan(segment).any() and segment_std > 1e-6: # Проверка на NaN и на то, что сигнал не постоянный
        try:
            # Используем медиану для порога высоты пика, чтобы быть устойчивее к выбросам
            segment_median = np.median(segment)
            # Можно добавить динамический порог prominence для большей робастности
            # prominence_threshold = 0.3 * segment_std
            # peaks, properties = find_peaks(segment, distance=min_dist, height=segment_median, prominence=prominence_threshold)
            peaks, _ = find_peaks(segment, distance=min_dist, height=segment_median)

            if len(peaks) > 1:
                # Рассчитываем ЧСС на основе интервалов между пиками для большей точности
                peak_intervals_sec = np.diff(peaks) / fs
                if len(peak_intervals_sec) > 0:
                    avg_interval = np.mean(peak_intervals_sec)
                    if avg_interval > 0:
                        hr_calc = 60.0 / avg_interval
                        if min_hr <= hr_calc <= max_hr:
                            hr = hr_calc
                # Запасной вариант: расчет по количеству пиков (менее точный)
                elif len(peaks) > 0:
                    actual_segment_duration_sec = len(segment) / fs
                    peaks_per_sec = len(peaks) / actual_segment_duration_sec if actual_segment_duration_sec > 0 else 0
                    hr_calc = peaks_per_sec * 60.0
                    if min_hr <= hr_calc <= max_hr:
                         hr = hr_calc

        except Exception as e:
            # print(f"Ошибка расчета ЧСС: {e}")
            pass # Возвращаем NaN, если произошла ошибка
    return hr

# --- БЛОК: Классические Методы (POS, CHROM, ICA) ---
def chrom_method(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0]<2 or rgb_buffer.shape[1]!=3: return np.array([])
    std_rgb=np.std(rgb_buffer,axis=0,keepdims=True); std_rgb[std_rgb<1e-8]=1.0; RGB_std=rgb_buffer/std_rgb
    X=3*RGB_std[:,0]-2*RGB_std[:,1]; Y=1.5*RGB_std[:,0]+RGB_std[:,1]-1.5*RGB_std[:,2]
    std_X=np.std(X); std_Y=np.std(Y); alpha=std_X/(std_Y+1e-8) if std_Y>1e-8 else 1.0
    chrom_signal=X-alpha*Y;
    try: return detrend(chrom_signal)
    except ValueError: return chrom_signal - np.mean(chrom_signal) # Центрирование, если detrend не удался

def pos_method(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0]<2 or rgb_buffer.shape[1]!=3: return np.array([])
    mean_rgb=np.mean(rgb_buffer,axis=0,keepdims=True); mean_rgb[mean_rgb<1e-8]=1.0; rgb_norm=rgb_buffer/mean_rgb
    try: rgb_detrended=detrend(rgb_norm,axis=0)
    except ValueError: rgb_detrended=rgb_norm-np.mean(rgb_norm,axis=0,keepdims=True) # Центрирование
    proj_mat=np.array([[0,1,-1],[-2,1,1]]); proj_sig=np.dot(rgb_detrended,proj_mat.T); std_dev=np.std(proj_sig,axis=0)
    alpha = 1.0 # Значение по умолчанию
    if std_dev.shape[0] >= 2 and std_dev[0] > 1e-8:
        alpha = std_dev[1] / std_dev[0]
    pos_signal=proj_sig[:,0]+alpha*proj_sig[:,1];
    try: return detrend(pos_signal)
    except ValueError: return pos_signal - np.mean(pos_signal) # Центрирование

def ica_method(rgb_buffer, fs_approx=30):
    if rgb_buffer is None or rgb_buffer.shape[0] < 3 or rgb_buffer.shape[1] != 3: return np.array([])
    try: rgb_detrended = detrend(rgb_buffer, axis=0)
    except ValueError: rgb_detrended = rgb_buffer - np.mean(rgb_buffer, axis=0, keepdims=True)

    # --- Проверка ранга и адаптация n_components ---
    n_components = 3
    try:
        cov_matrix = np.cov(rgb_detrended.T)
        rank = np.linalg.matrix_rank(cov_matrix)
        n_components = max(1, min(3, rank)) # Ограничиваем 1 <= n <= rank <= 3

        # Если ранг = 1, ICA бессмысленен, выбираем лучший канал по дисперсии
        if n_components == 1:
            stds = np.std(rgb_detrended, axis=0)
            if len(stds) > 0 and np.max(stds) > 1e-8:
                best_channel_idx = np.argmax(stds)
                best_channel = rgb_detrended[:, best_channel_idx]
                try: return detrend(best_channel)
                except ValueError: return best_channel - np.mean(best_channel)
            else: # Если все каналы почти нулевые
                return np.zeros(rgb_buffer.shape[0]) # Возвращаем нули
    except np.linalg.LinAlgError:
        print("Предупреждение: Ошибка при расчете ранга ковариационной матрицы в ICA.")
        # Продолжаем с n_components=3, ICA может выдать ошибку ниже
        pass
    except Exception as e_rank:
        print(f"Неожиданная ошибка при расчете ранга в ICA: {e_rank}")
        pass

    if n_components < 2: # Если после проверки ранга остался только 1 компонент
       # Мы уже обработали этот случай выше, но на всякий случай
       stds = np.std(rgb_detrended, axis=0)
       if len(stds) > 0 and np.max(stds) > 1e-8:
           best_channel = rgb_detrended[:, np.argmax(stds)]
           try: return detrend(best_channel)
           except ValueError: return best_channel - np.mean(best_channel)
       else:
           return np.zeros(rgb_buffer.shape[0])

    # --- Выполнение ICA ---
    S_ = np.array([])
    try:
        # Используем 'unit-variance' для стандартизации перед ICA
        ica = FastICA(n_components=n_components, random_state=42, max_iter=300, tol=0.05, whiten='unit-variance')
        S_ = ica.fit_transform(rgb_detrended)
        if S_.shape[1] != n_components:
            print(f"Предупреждение ICA: Ожидалось {n_components} компонент, получено {S_.shape[1]}.")
            return np.array([]) # Неожиданный результат
    except ValueError as e_ica:
        print(f"Ошибка ValueError в FastICA (возможно, из-за данных): {e_ica}")
        return np.array([])
    except Exception as e_ica_general:
        print(f"Неожиданная ошибка в FastICA: {e_ica_general}")
        traceback.print_exc()
        return np.array([])

    if S_.size == 0: return np.array([]) # ICA не вернула компоненты

    # --- Выбор лучшего компонента по спектральной мощности ---
    best_idx=-1; max_power=-1; low_f=BANDPASS_LOW; high_f=BANDPASS_HIGH
    for i in range(S_.shape[1]):
        sig=S_[:,i]
        # Проверяем валидность сигнала компонента
        if len(sig)<2 or np.std(sig)<1e-8 or np.isnan(sig).any(): continue
        try:
            # Используем окно Ханна для уменьшения краевых эффектов FFT
            fft_win=sig*np.hanning(len(sig)); fft_v=np.fft.rfft(fft_win)
            # Рассчитываем частоты для rfft
            fs_eff = fs_approx if fs_approx > 0 else FPS_ESTIMATED
            fft_f=np.fft.rfftfreq(len(sig), 1.0 / fs_eff)
            # Мощность = квадрат амплитуды
            p_s=np.abs(fft_v)**2;
            # Маска для интересующего диапазона частот (пульса)
            mask=(fft_f>=low_f)&(fft_f<=high_f)
            # Считаем среднюю мощность в диапазоне
            if np.any(mask): # Убедимся, что в диапазоне есть частоты
                p_band=np.mean(p_s[mask])
                if p_band > max_power:
                    max_power=p_band; best_idx=i
        except Exception as e_fft:
            # print(f"Ошибка при расчете FFT для компонента {i} в ICA: {e_fft}")
            continue # Пропускаем этот компонент

    # Если не удалось выбрать по FFT (например, все сигналы плохие или ошибки)
    if best_idx == -1:
        stds = np.std(S_,axis=0)
        valid_stds = stds[~np.isnan(stds)] # Игнорируем NaN дисперсии
        if len(valid_stds) > 0 and np.max(valid_stds) > 1e-8:
            # Находим индекс максимальной валидной дисперсии в исходном массиве stds
            best_idx = np.nanargmax(stds)
            print("Предупреждение ICA: Выбор компонента по дисперсии, а не по FFT.")
        else:
            # print("Предупреждение ICA: Не удалось выбрать компонент ни по FFT, ни по дисперсии.")
            return np.array([]) # Не можем выбрать компонент

    selected_component = S_[:, best_idx]
    # Финальное детрендирование или центрирование выбранного компонента
    try: return detrend(selected_component)
    except ValueError: return selected_component - np.mean(selected_component)

# --- БЛОК: Определение Моделей Нейронных Сетей ---
# (Без изменений - сами классы моделей)
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

def get_model_instance(model_name, model_path):
    """Создает экземпляр модели и загружает веса."""
    model_class_map = {
        "SimplePhysNet": SimplePhysNet, "ImprovedPhysNet": ImprovedPhysNet,
        "HAFNet": HAFNet, "PhysFormer": PhysFormer
    }
    model_class = model_class_map.get(model_name)
    if not model_class:
        print(f"Предупреждение: Неизвестный класс модели для {model_name}")
        return None

    print(f"  Попытка создания модели: {model_name}...")
    model = model_class().to(device) # Создаем модель на нужном устройстве
    print(f"  Попытка загрузки весов из: {model_path} на {device}...")
    try:
        # Важно: Используем map_location=device при загрузке!
        state_dict = torch.load(model_path, map_location=device)

        # Фильтрация ключей, если state_dict сохранен с 'module.'
        original_keys = list(state_dict.keys())
        if all(k.startswith('module.') for k in original_keys):
             print("  Обнаружен префикс 'module.' в state_dict, удаляем его.")
             state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        model_dict = model.state_dict()
        # Фильтруем state_dict, чтобы оставить только совпадающие по имени и размеру ключи
        state_dict_filtered = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

        if not state_dict_filtered:
             print(f"  ПРЕДУПРЕЖДЕНИЕ: Не найдено совпадающих слоев в state_dict для {model_name} из {model_path}")
             # Можно попытаться загрузить все равно с strict=False, но это рискованно
             # model.load_state_dict(state_dict, strict=False) # Раскомментируйте на свой страх и риск
             return None # Безопаснее считать, что загрузка не удалась

        print(f"  Найдено {len(state_dict_filtered)} совпадающих слоев для загрузки.")

        # Обновляем словарь модели и загружаем с strict=False, так как мы уже отфильтровали
        model_dict.update(state_dict_filtered)
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        if missing_keys:
            print(f"  Предупреждение: Некоторые слои модели НЕ были загружены (отсутствуют в файле): {missing_keys}")
        if unexpected_keys:
             print(f"  Предупреждение: Некоторые ключи из файла НЕ были использованы (нет в модели): {unexpected_keys}")

        model.eval() # Переводим модель в режим оценки
        print(f"  УСПЕШНО загружена модель: {model_name} на {device}")
        return model
    except FileNotFoundError:
        st.error(f"Ошибка: Файл модели не найден - {model_path}. Проверьте путь и наличие файла.")
        print(f"  ОШИБКА: Файл модели не найден - {model_path}")
        return None
    except Exception as e:
        st.error(f"Ошибка загрузки весов для {model_name} из {model_path}: {e}")
        print(f"  ОШИБКА загрузки весов для {model_name} из {model_path}: {e}")
        traceback.print_exc()
        return None

# --- Функция Загрузки Моделей НС и Детектора с Кешированием ---
@st.cache_resource # Кешируем ресурсы для всего сеанса пользователя
def load_resources():
    """Загружает модели НС и детектор лиц MediaPipe."""
    models = {}
    print("\n--- Загрузка моделей НС ---")
    # Проверяем доступность файлов ДО попытки загрузки
    available_models = {}
    for name, path in NN_MODEL_PATHS.items():
        if path and os.path.exists(path):
             available_models[name] = path
             print(f"Файл модели {name} найден: {path}")
        elif path:
             print(f"ПРЕДУПРЕЖДЕНИЕ: Файл модели {name} НЕ НАЙДЕН по пути: {path}")
        else:
             print(f"ПРЕДУПРЕЖДЕНИЕ: Путь для модели {name} не указан.")

    if not available_models:
         st.warning("Не найдено ни одного файла модели нейронной сети. Будут доступны только классические методы.", icon="⚠️")

    # Загружаем только доступные модели
    for name, path in available_models.items():
        model_instance = get_model_instance(name, path)
        if model_instance:
            models[name] = model_instance
        else:
             # Модель не загрузилась, сообщаем об этом
             st.warning(f"Модель {name} не была загружена из-за ошибки.", icon="❌")


    print(f"--- Загружено {len(models)} НС моделей ---")

    print("\n--- Загрузка MediaPipe Face Detection ---")
    face_detector = None
    try:
        mp_face_detection = mp.solutions.face_detection
        # model_selection=0 для коротких дистанций (<2m), =1 для дальних (<5m)
        face_detector = mp_face_detection.FaceDetection(min_detection_confidence=MIN_DETECTION_CONFIDENCE, model_selection=0)
        print("MediaPipe Face Detection успешно загружен.")
    except Exception as e:
        print(f"Критическая ошибка инициализации MediaPipe: {e}")
        traceback.print_exc()
        # Если MediaPipe не загрузился, приложение не сможет работать
        st.error(f"Не удалось загрузить MediaPipe Face Detection: {e}. Приложение не может продолжить.", icon="🚨")
        # Не вызываем st.stop() здесь, вернем None и обработаем в run_streamlit_app
        return models, None # Возвращаем None для детектора

    return models, face_detector

# --- Класс Видеопроцессора для Streamlit-WebRTC ---
class RPPGVideoProcessor(VideoProcessorBase):
    def __init__(self, nn_models, face_detector):
        print(">>> RPPGVideoProcessor: Инициализация...")
        # --- Ресурсы ---
        if face_detector is None:
             # Эта проверка должна быть ДО присваивания self._face_detector
             print("ОШИБКА: Детектор лиц не передан в RPPGVideoProcessor!")
             raise ValueError("Face detector is required for RPPGVideoProcessor initialization.")
        self._nn_models = nn_models if nn_models is not None else {}
        self._face_detector = face_detector
        self._available_methods = ["POS", "CHROM", "ICA"] + sorted(list(self._nn_models.keys()))
        self._selected_method = self._available_methods[0] if self._available_methods else None # Метод по умолчанию или None

        # --- Буферы и Состояние ---
        # Уменьшаем размер буфера для классики, чтобы быстрее реагировать
        CLASSIC_BUFFER_SIZE = int(BUFFER_SIZE * 0.7) # ~7 секунд при 30 FPS
        self._mean_rgb_buffer = collections.deque(maxlen=CLASSIC_BUFFER_SIZE)
        self._rppg_signal_buffer = collections.deque(maxlen=BUFFER_SIZE) # Буфер для rPPG сигнала (для сглаживания и графиков)
        self._timestamps = collections.deque(maxlen=CLASSIC_BUFFER_SIZE)
        self._nn_frame_buffer = collections.deque(maxlen=NN_WINDOW_SIZE_FRAMES) # Буфер кадров для НС

        self._current_hr = np.nan
        self._smoothed_hr = np.nan
        self._calculated_fps = FPS_ESTIMATED
        self._last_fps_time = time.monotonic() # Используем monotonic для измерения интервалов
        self._frame_count_for_fps = 0
        self._last_signal_time = time.monotonic() # Время последнего успешного ROI
        self._nn_ready_to_predict = False
        self._face_found_prev_frame = False # Отслеживаем предыдущее состояние
        self._status_message = "Инициализация..."
        self._plot_data = np.array([]) # Данные для графика

        # --- Блокировки для Потокобезопасности ---
        self._buffer_lock = threading.Lock() # Защищает все буферы (_mean_rgb, _timestamps, _nn_frame, _rppg_signal)
        self._hr_lock = threading.Lock()     # Защищает _current_hr и _smoothed_hr
        self._plot_lock = threading.Lock()   # Защищает _plot_data
        self._status_lock = threading.Lock() # Защищает _status_message
        self._method_lock = threading.Lock() # Защищает _selected_method и _nn_ready_to_predict

        if self._selected_method is None:
             print("ОШИБКА: Нет доступных методов rPPG!")
             self._update_status("Ошибка: Нет доступных методов.")
        else:
             print(f"RPPGVideoProcessor инициализирован. Доступные методы: {self._available_methods}. Текущий: {self._selected_method}")

    def _update_status(self, message):
        with self._status_lock:
            # Обновляем только если сообщение изменилось, чтобы не спамить лог/UI
            if self._status_message != message:
                self._status_message = message
                # print(f"Status update: {message}") # Для отладки

    def get_status(self):
        with self._status_lock:
            return self._status_message

    def get_hr(self):
        with self._hr_lock:
            # Возвращаем копию на всякий случай (хотя для float это не так критично)
            return self._smoothed_hr

    def get_fps(self):
        # FPS рассчитывается внутри recv, просто возвращаем последнее значение
        # Не требует блокировки, т.к. читается одним потоком, пишется другим,
        # но чтение атомарно для float/int в Python.
        return self._calculated_fps

    def get_plot_data(self):
        with self._plot_lock:
            # Возвращаем копию, чтобы избежать проблем с изменением во время чтения
            return self._plot_data.copy()

    def set_method(self, method_name):
         if method_name in self._available_methods:
              with self._method_lock:
                   if self._selected_method != method_name:
                        print(f"Метод изменен с '{self._selected_method}' на '{method_name}'")
                        self._selected_method = method_name
                        # Сброс состояния при смене метода
                        with self._buffer_lock:
                            self._mean_rgb_buffer.clear()
                            self._rppg_signal_buffer.clear()
                            self._timestamps.clear()
                            self._nn_frame_buffer.clear()
                        with self._hr_lock:
                            self._current_hr = np.nan
                            self._smoothed_hr = np.nan
                        with self._plot_lock:
                            self._plot_data = np.array([])
                        self._nn_ready_to_predict = False
                        self._update_status(f"Метод изменен на {method_name}. Ожидание данных...")
                        print(">>> Буферы и состояние сброшены из-за смены метода.")
         else:
              print(f"Предупреждение: Попытка установить неверный метод '{method_name}'")

    # Основной метод обработки кадров
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            current_time_monotonic = time.monotonic() # Для FPS и интервалов
            current_time_perf = time.perf_counter() # Для точного измерения времени операций

            # Преобразование кадра из av.VideoFrame в BGR NumPy массив
            image = frame.to_ndarray(format="bgr24")
            if image is None or image.size == 0:
                # print("Пустой кадр получен")
                return frame # Возвращаем оригинал, если кадр пуст

            height, width, _ = image.shape
            roi_display = image.copy() # Копия для рисования ROI

            # --- Расчет FPS ---
            self._frame_count_for_fps += 1
            elapsed_time = current_time_monotonic - self._last_fps_time
            if elapsed_time >= 1.0: # Обновляем раз в секунду
                self._calculated_fps = self._frame_count_for_fps / elapsed_time
                # Ограничиваем FPS разумными пределами
                self._calculated_fps = max(1.0, min(self._calculated_fps, 60.0))
                self._frame_count_for_fps = 0
                self._last_fps_time = current_time_monotonic
                # print(f"Calculated FPS: {self._calculated_fps:.1f}") # Debug FPS

            actual_fps = self._calculated_fps # Используем рассчитанный FPS для расчетов ЧСС/фильтрации

            # --- Получение текущего метода (потокобезопасно) ---
            with self._method_lock:
                current_method = self._selected_method
                # Проверка на случай, если метод не был установлен при инициализации
                if current_method is None:
                     self._update_status("Ошибка: Метод не выбран.")
                     # Возвращаем кадр без обработки, если нет метода
                     return av.VideoFrame.from_ndarray(roi_display, format="bgr24")

            # --- Обработка Кадра (MediaPipe) ---
            face_detection_start = time.perf_counter()
            # Конвертация в RGB для MediaPipe
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False # Оптимизация
            results = self._face_detector.process(frame_rgb)
            face_detection_duration = time.perf_counter() - face_detection_start

            face_found_this_frame = False
            valid_roi_extracted = False

            if results.detections:
                # Сортируем по уверенности и берем самый уверенный результат
                best_detection = max(results.detections, key=lambda x: x.score[0])

                if best_detection.score[0] >= MIN_DETECTION_CONFIDENCE:
                    face_found_this_frame = True
                    bboxC = best_detection.location_data.relative_bounding_box
                    if bboxC: # Убедимся, что bounding box существует
                        # Рассчитываем координаты ROI с паддингом
                        xmin=int(bboxC.xmin*width); ymin=int(bboxC.ymin*height)
                        w=int(bboxC.width*width); h=int(bboxC.height*height)
                        pad_w=int(w*ROI_PADDING_FACTOR); pad_h=int(h*ROI_PADDING_FACTOR)
                        # Убедимся, что координаты не выходят за пределы кадра
                        x1=max(0,xmin-pad_w); y1=max(0,ymin-pad_h)
                        x2=min(width,xmin+w+pad_w); y2=min(height,ymin+h+pad_h)

                        # Проверяем, что ROI имеет ненулевые размеры
                        if y2 > y1 and x2 > x1:
                            valid_roi_extracted = True
                            roi = image[y1:y2, x1:x2] # Используем BGR из исходного кадра
                            # Рисуем зеленый прямоугольник на копии кадра
                            cv2.rectangle(roi_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # --- Обновление буферов (с блокировкой) ---
                            # Блокируем доступ к буферам на минимальное время
                            with self._buffer_lock:
                                self._timestamps.append(current_time_monotonic)
                                # 1. Средний цвет для классики (BGR -> RGB)
                                mean_bgr_roi = np.mean(roi, axis=(0, 1))
                                self._mean_rgb_buffer.append(mean_bgr_roi[::-1])

                                # 2. Предобработанный ROI для НС (только если выбран метод НС)
                                with self._method_lock: # Быстрая проверка метода внутри блока буфера
                                     is_nn_method = self._selected_method in self._nn_models

                                if is_nn_method:
                                    try:
                                        # Изменение размера и нормализация для НС
                                        roi_resized_nn = cv2.resize(roi, NN_RESIZE_DIM, interpolation=cv2.INTER_AREA) # INTER_AREA лучше для уменьшения
                                        # Нормализация в диапазон [-1, 1]
                                        roi_norm = (roi_resized_nn.astype(np.float32) / 127.5) - 1.0
                                        # Транспонирование в формат (C, H, W) для PyTorch
                                        self._nn_frame_buffer.append(roi_norm.transpose(2, 0, 1))
                                    except cv2.error as e_resize:
                                        print(f"Ошибка cv2.resize/обработки для НС: {e_resize}")
                                        # Пропускаем добавление этого кадра в буфер НС
                                        pass
                                    except Exception as e_nn_prep:
                                         print(f"Неожиданная ошибка при подготовке кадра для НС: {e_nn_prep}")
                                         pass

                            # Обновляем время последнего успешного обнаружения лица и ROI
                            self._last_signal_time = current_time_monotonic


            # --- Управление состоянием и статусом ---
            if valid_roi_extracted: # Если лицо найдено И ROI валидный
                with self._method_lock: # Блокировка для проверки метода и _nn_ready_to_predict
                     if self._selected_method in self._nn_models:
                          # Проверяем и обновляем флаг готовности НС под блокировкой
                          nn_buffer_len = len(self._nn_frame_buffer) # Получаем длину внутри блока buffer_lock
                          if nn_buffer_len == NN_WINDOW_SIZE_FRAMES:
                               if not self._nn_ready_to_predict: # Обновляем статус только при переходе
                                    self._nn_ready_to_predict = True
                                    self._update_status(f"Буфер {self._selected_method} готов к предсказанию!")
                          else:
                               # Если буфер еще не полон, сбрасываем флаг и показываем прогресс
                               self._nn_ready_to_predict = False
                               self._update_status(f"Сбор данных для {self._selected_method}: {nn_buffer_len}/{NN_WINDOW_SIZE_FRAMES}...")
                     else: # Классический метод
                          self._update_status("Обработка...") # Общий статус
                self._face_found_prev_frame = True

            else: # Если лицо не найдено или ROI невалидный
                if self._face_found_prev_frame: # Если лицо только что потерялось
                     print("Лицо потеряно.")
                     # Сбрасываем готовность НС, если был выбран метод НС
                     with self._method_lock:
                          if self._selected_method in self._nn_models:
                               with self._buffer_lock:
                                    self._nn_frame_buffer.clear() # Очищаем буфер НС
                               self._nn_ready_to_predict = False
                               print(">>> Буфер НС очищен из-за потери лица.")
                     # Сбрасываем ЧСС, т.к. сигнал прервался
                     with self._hr_lock:
                          self._current_hr = np.nan
                          self._smoothed_hr = np.nan
                     with self._plot_lock:
                          self._plot_data = np.array([])

                # Показываем сообщение, если лицо не видно дольше ~1.5 секунды
                if current_time_monotonic - self._last_signal_time > 1.5:
                    self._update_status("Лицо не обнаружено...")

                self._face_found_prev_frame = False


            # --- Расчет rPPG Сигнала и ЧСС ---
            processing_start_time = time.perf_counter()
            calculated_hr_this_step = np.nan
            signal_processed_success = False
            current_rppg_signal_for_plot = None

            # Выбираем, какой метод использовать
            with self._method_lock: current_method_local = self._selected_method

            # --- Классические методы ---
            if current_method_local in ["POS", "CHROM", "ICA"]:
                # Требуем достаточно данных для анализа (~2 секунды)
                min_len_classical = int(actual_fps * 2.0)

                with self._buffer_lock: # Доступ к буферу RGB
                    rgb_buffer_len = len(self._mean_rgb_buffer)
                    if rgb_buffer_len >= min_len_classical:
                        rgb_data = np.array(self._mean_rgb_buffer)
                        ts_data = np.array(self._timestamps)
                    else:
                        rgb_data = None # Недостаточно данных
                        ts_data = None

                if rgb_data is not None and ts_data is not None and len(ts_data) > 1:
                    # Оцениваем реальный FPS по временным меткам
                    fs_real = len(ts_data) / (ts_data[-1] - ts_data[0]) if (ts_data[-1] - ts_data[0]) > 0 else actual_fps
                    fs_real = max(5.0, min(fs_real, 60.0)) # Ограничение разумными значениями

                    rppg_signal_raw = np.array([])
                    method_execution_success = False
                    try:
                        if current_method_local == "POS":
                            rppg_signal_raw = pos_method(rgb_data)
                        elif current_method_local == "CHROM":
                            rppg_signal_raw = chrom_method(rgb_data)
                        elif current_method_local == "ICA":
                            # Передаем более точный FPS в ICA
                            rppg_signal_raw = ica_method(rgb_data, fs_real)
                        method_execution_success = (rppg_signal_raw is not None and rppg_signal_raw.size > 1)
                    except Exception as e_method:
                        print(f"Ошибка выполнения метода {current_method_local}: {e_method}")
                        # traceback.print_exc() # Раскомментировать для детальной отладки

                    if method_execution_success:
                        # Нормализация и фильтрация
                        rppg_norm = normalize_signal_np(rppg_signal_raw)
                        rppg_filt = bandpass_filter(rppg_norm, fs_real)

                        if rppg_filt is not None and rppg_filt.size > 1:
                             with self._buffer_lock: # Обновляем ОБЩИЙ буфер rPPG сигнала
                                  # Добавляем только новые точки, чтобы избежать дублирования
                                  current_rppg_len = len(self._rppg_signal_buffer)
                                  num_new_points = len(rppg_filt) - current_rppg_len
                                  if num_new_points > 0:
                                       self._rppg_signal_buffer.extend(rppg_filt[-num_new_points:])
                                  elif current_rppg_len == 0 and len(rppg_filt) > 0: # Если буфер был пуст
                                       self._rppg_signal_buffer.extend(rppg_filt)

                                  # Данные для графика - из ОБЩЕГО буфера rPPG
                                  if len(self._rppg_signal_buffer) > 0:
                                       signal_processed_success = True
                                       current_rppg_signal_for_plot = np.array(self._rppg_signal_buffer)

                             # Расчет ЧСС (используем сигнал ИЗ ОБЩЕГО БУФЕРА)
                             min_len_hr = int(HR_WINDOW_SEC_ANALYSIS * fs_real * 0.8) # Требуем 80% окна для расчета
                             if signal_processed_success and len(current_rppg_signal_for_plot) >= min_len_hr:
                                 # Передаем сигнал и реальный FPS
                                 hr_val = calculate_hr(current_rppg_signal_for_plot, fs_real, window_sec=HR_WINDOW_SEC_ANALYSIS)
                                 if not np.isnan(hr_val):
                                     calculated_hr_this_step = hr_val
                                 # else: print("HR calculation returned NaN") # Debug

            # --- Нейросетевые методы ---
            elif current_method_local in self._nn_models:
                 nn_should_predict = False
                 with self._method_lock: # Проверяем флаг готовности под блокировкой
                      if self._nn_ready_to_predict:
                           nn_should_predict = True
                           self._nn_ready_to_predict = False # Сбрасываем флаг СРАЗУ, чтобы не запустить предсказание дважды

                 if nn_should_predict:
                    nn_input_numpy = None
                    with self._buffer_lock: # Копируем буфер НС под блокировкой
                         if len(self._nn_frame_buffer) == NN_WINDOW_SIZE_FRAMES: # Доп. проверка
                              nn_input_numpy = np.array(self._nn_frame_buffer)
                              # Важно: Очищаем буфер СРАЗУ после копирования, чтобы начать копить заново
                              self._nn_frame_buffer.clear()
                              print(f">>> Буфер НС ({current_method_local}) скопирован ({nn_input_numpy.shape}) и очищен для следующего предсказания.")
                         else:
                              # Этого не должно произойти из-за флага _nn_ready_to_predict, но на всякий случай
                              print(f"ПРЕДУПРЕЖДЕНИЕ: Попытка предсказания НС, но буфер не полон ({len(self._nn_frame_buffer)}/{NN_WINDOW_SIZE_FRAMES}).")


                    if nn_input_numpy is not None:
                        nn_predict_start_time = time.perf_counter()
                        rppg_signal_raw_nn = np.array([])
                        try:
                            model_instance = self._nn_models[current_method_local]
                            # Формируем тензор: (T, C, H, W) -> (B=1, C, T, H, W)
                            # Переставляем оси: T и C меняются местами
                            nn_input_permuted = nn_input_numpy.transpose(1, 0, 2, 3)
                            input_tensor = torch.from_numpy(nn_input_permuted).float().unsqueeze(0).to(device)

                            with torch.no_grad(): # Отключаем расчет градиентов
                                rppg_signal_raw_nn = model_instance(input_tensor).squeeze().cpu().numpy()

                            nn_predict_duration = time.perf_counter() - nn_predict_start_time
                            # print(f"NN ({current_method_local}) prediction time: {nn_predict_duration:.3f}s")

                        except Exception as e_nn_pred:
                            print(f"Ошибка предсказания НС {current_method_local}: {e_nn_pred}")
                            traceback.print_exc()
                            # Сбрасываем ЧСС в случае ошибки предсказания
                            with self._hr_lock:
                                 self._current_hr = np.nan
                                 self._smoothed_hr = np.nan

                        # --- Обработка выхода НС и расчет ЧСС ---
                        if rppg_signal_raw_nn is not None and rppg_signal_raw_nn.size > 1:
                            # Выход НС уже должен быть временным рядом (длиной NN_WINDOW_SIZE_FRAMES)
                            rppg_norm_nn = normalize_signal_np(rppg_signal_raw_nn)
                            # Фильтруем выход НС, используя оценочный FPS, т.к. временные метки кадров НС мы не хранили явно
                            rppg_filt_nn = bandpass_filter(rppg_norm_nn, actual_fps)

                            if rppg_filt_nn is not None and rppg_filt_nn.size > 1:
                                signal_processed_success = True
                                # Для НС сигнал для графика - это ПОСЛЕДНИЙ предсказанный и отфильтрованный сигнал
                                current_rppg_signal_for_plot = rppg_filt_nn

                                # Добавляем результат НС в ОБЩИЙ буфер rPPG для консистентности (хотя он может не использоваться напрямую для ЧСС)
                                with self._buffer_lock:
                                     self._rppg_signal_buffer.clear() # Очищаем старые данные классики/предыдущего НС
                                     self._rppg_signal_buffer.extend(rppg_filt_nn)


                                # Расчет ЧСС непосредственно из выхода НС
                                # Используем все окно предсказания НС для анализа
                                nn_window_duration_sec = len(rppg_filt_nn) / actual_fps if actual_fps > 0 else NN_WINDOW_SIZE_FRAMES / FPS_ESTIMATED
                                analysis_win_sec_nn = nn_window_duration_sec # Анализируем весь выход НС
                                min_samples_nn_hr = int(actual_fps * 1.5) # Требуем хотя бы ~1.5с сигнала

                                if len(rppg_filt_nn) >= min_samples_nn_hr:
                                    # Передаем сигнал НС и актуальный FPS
                                    hr_val = calculate_hr(rppg_filt_nn, actual_fps, window_sec=analysis_win_sec_nn)
                                    if not np.isnan(hr_val):
                                        calculated_hr_this_step = hr_val
                                    # else: print("HR calculation (NN) returned NaN") # Debug

            # --- Сглаживание и Обновление состояния ЧСС (потокобезопасно) ---
            with self._hr_lock:
                 if not np.isnan(calculated_hr_this_step):
                     # Если предыдущее значение было NaN, инициализируем оба значения
                     if np.isnan(self._smoothed_hr):
                          self._current_hr = calculated_hr_this_step
                          self._smoothed_hr = calculated_hr_this_step
                     else: # EMA сглаживание
                          self._current_hr = calculated_hr_this_step # Сохраняем последнее "сырое"
                          # Применяем EMA
                          self._smoothed_hr = (HR_SMOOTHING_FACTOR * calculated_hr_this_step +
                                             (1 - HR_SMOOTHING_FACTOR) * self._smoothed_hr)
                     # print(f"HR updated: Raw={self._current_hr:.1f}, Smoothed={self._smoothed_hr:.1f}") # Debug HR
                 # Если calculated_hr_this_step is NaN (лицо потеряно, ошибка расчета и т.д.),
                 # НЕ обновляем _smoothed_hr, он сохраняет последнее валидное значение.
                 # Можно добавить логику сброса smoothed_hr в NaN, если лицо потеряно слишком долго.
                 elif not self._face_found_prev_frame and (current_time_monotonic - self._last_signal_time > 5.0): # Сброс через 5 сек без лица
                      if not np.isnan(self._smoothed_hr): print("Сброс ЧСС из-за долгого отсутствия лица.")
                      self._current_hr = np.nan
                      self._smoothed_hr = np.nan


            # --- Обновление данных для графика (потокобезопасно) ---
            # Обновляем, только если сигнал был успешно обработан в этом шаге
            if signal_processed_success and current_rppg_signal_for_plot is not None:
                 with self._plot_lock:
                      # Ограничиваем размер данных для графика последними BUFFER_SIZE точками
                      self._plot_data = current_rppg_signal_for_plot[-BUFFER_SIZE:]

            processing_duration = time.perf_counter() - processing_start_time
            total_frame_time = time.perf_counter() - current_time_perf
            # print(f"Frame timing: Total={total_frame_time*1000:.1f}ms, FaceDet={face_detection_duration*1000:.1f}ms, Processing={processing_duration*1000:.1f}ms") # Debug timing

            # Возвращаем кадр с нарисованным ROI (если найден) обратно в WebRTC
            return av.VideoFrame.from_ndarray(roi_display, format="bgr24")

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА в RPPGVideoProcessor.recv: {e}")
            traceback.print_exc()
            # В случае ошибки пытаемся вернуть исходный кадр, чтобы поток не прервался
            try:
                # Создаем пустой кадр того же размера, если исходный frame недоступен
                if 'image' in locals() and image is not None:
                     blank_frame = np.zeros_like(image)
                     return av.VideoFrame.from_ndarray(blank_frame, format="bgr24")
                else:
                     # Если даже размер неизвестен, возвращаем сам frame (может быть None)
                     return frame
            except Exception as e_fallback:
                 print(f"Ошибка при возврате кадра после исключения в recv: {e_fallback}")
                 return None # Крайний случай


# --- Основная Функция Приложения Streamlit ---
def run_streamlit_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.markdown("Используйте вебкамеру для измерения пульса в реальном времени с помощью различных rPPG методов.")

    # --- Загрузка Ресурсов (один раз для сессии) ---
    # Используем session_state для хранения ресурсов после первой загрузки
    if "resources_loaded" not in st.session_state:
        st.session_state.resources_loaded = False
        st.session_state.loaded_nn_models = None
        st.session_state.face_detector = None

    if not st.session_state.resources_loaded:
        print(">>> Загрузка ресурсов...")
        with st.spinner("Загрузка моделей и детектора лиц... Пожалуйста, подождите."):
            try:
                loaded_nn_models, face_detector = load_resources()
                st.session_state.loaded_nn_models = loaded_nn_models
                st.session_state.face_detector = face_detector
                st.session_state.resources_loaded = True
                print(">>> Ресурсы успешно загружены.")
            except Exception as e:
                st.error(f"Не удалось загрузить необходимые ресурсы: {e}. Приложение остановлено.", icon="🚨")
                traceback.print_exc()
                st.session_state.resources_loaded = False # Помечаем, что загрузка не удалась
                st.stop() # Останавливаем выполнение скрипта
    else:
        # Получаем ресурсы из session_state при последующих запусках
        loaded_nn_models = st.session_state.loaded_nn_models
        face_detector = st.session_state.face_detector
        print(">>> Ресурсы взяты из session_state.")

    # --- Проверка критически важного детектора лиц ---
    if face_detector is None:
         # Эта ситуация может возникнуть, если load_resources вернул None для детектора
         st.error("Критическая ошибка: Детектор лиц MediaPipe не был загружен. Приложение не может работать.", icon="🆘")
         st.stop()

    # Формируем список доступных методов ПОСЛЕ загрузки
    available_methods = ["POS", "CHROM", "ICA"] + sorted(list(loaded_nn_models.keys()))
    if not available_methods:
         st.error("Ошибка: Не доступно ни одного метода rPPG (ни классического, ни НС).", icon="❌")
         st.stop() # Нет смысла продолжать без методов

    # --- Настройки в Сайдбаре ---
    st.sidebar.header("Настройки")
    # Используем ключ для сохранения выбора пользователя между перезапусками
    selected_method = st.sidebar.selectbox(
        "Выберите метод измерения:",
        available_methods,
        index=0, # Начинаем с первого доступного метода
        key="method_select"
    )

    # --- Инициализация Видеопроцессора в session_state (УЛУЧШЕННАЯ) ---
    # Инициализируем, только если ресурсы загружены и процессор еще не создан
    if st.session_state.resources_loaded and "rppg_processor" not in st.session_state:
        print(">>> Попытка создания нового экземпляра RPPGVideoProcessor...")
        try:
            # Передаем загруженные ресурсы в конструктор
            processor_instance = RPPGVideoProcessor(loaded_nn_models, face_detector)
            # Сохраняем в session_state ТОЛЬКО если создание успешно
            st.session_state.rppg_processor = processor_instance
            print(">>> Экземпляр RPPGVideoProcessor УСПЕШНО создан и добавлен в session_state.")
            # Устанавливаем выбранный метод сразу после создания
            st.session_state.rppg_processor.set_method(selected_method)
            print(f">>> Начальный метод '{selected_method}' установлен в процессоре.")
        except Exception as e:
            st.error(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации RPPGVideoProcessor: {e}", icon="🔥")
            st.error("Приложение не может продолжить работу.")
            traceback.print_exc()
            # Удаляем ключ, если инициализация не удалась, чтобы попытаться снова при перезапуске
            if "rppg_processor" in st.session_state:
                 del st.session_state.rppg_processor
            st.stop() # Останавливаем скрипт

    # --- Проверка наличия процессора перед использованием ---
    if "rppg_processor" not in st.session_state:
        # Эта ситуация возможна, если произошла ошибка выше или ресурсы не загрузились
        st.error("Ошибка: Видеопроцессор не был инициализирован.", icon="🚫")
        # Попробуем показать кнопку для перезагрузки страницы
        if st.button("Перезагрузить страницу"):
             st.rerun()
        st.stop()
    else:
        # Если процессор существует, обновляем его метод, если он изменился в UI
        # (set_method вызывается внутри, если метод действительно изменился)
        try:
            st.session_state.rppg_processor.set_method(selected_method)
        except Exception as e_set_method:
             st.error(f"Ошибка при обновлении метода в процессоре: {e_set_method}")
             traceback.print_exc()
             # Не останавливаем приложение, но сообщаем об ошибке


    # --- Настройки WebRTC (без ClientSettings) ---
    RTC_CONFIGURATION = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Стандартный STUN сервер Google
    }
    MEDIA_STREAM_CONSTRAINTS = {
        "video": {
            "frameRate": {"ideal": FPS_ESTIMATED, "min": 15}, # Запрашиваем желаемую частоту кадров
            # Можно добавить разрешение, если нужно, например:
            # "width": {"ideal": 640},
            # "height": {"ideal": 480}
        },
        "audio": False, # Аудио не нужно
    }

    # --- Запуск WebRTC Стримера ---
    print(">>> Попытка запуска webrtc_streamer...")
    webrtc_ctx = None # Инициализируем перед try
    try:
        # Фабрика теперь просто возвращает существующий экземпляр из session_state
        video_processor_factory = lambda: st.session_state.rppg_processor

        webrtc_ctx = webrtc_streamer(
            key="rppg-stream", # Уникальный ключ для компонента
            mode=WebRtcMode.SENDRECV, # Отправляем видео с клиента, получаем обработанное
            rtc_configuration=RTC_CONFIGURATION, # Передаем RTC config
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS, # Передаем media constraints
            video_processor_factory=video_processor_factory, # Используем лямбду
            async_processing=True, # Обработка в отдельном потоке, чтобы не блокировать UI
            # Атрибуты для видеоэлемента HTML
            video_html_attrs={
                 "style": "width: 100%; height: auto; max-width: 720px; border: 1px solid #ccc;",
                 "controls": False,
                 "autoPlay": True,
                 "muted": True # Важно для автовоспроизведения в некоторых браузерах
            }
        )
        print(">>> webrtc_streamer запущен.")
    except Exception as e_streamer:
        st.error(f"Ошибка при запуске или работе компонента webrtc_streamer: {e_streamer}", icon="🔌")
        traceback.print_exc()
        st.info("Попробуйте перезагрузить страницу или проверить разрешения камеры в браузере.")
        # Не вызываем st.stop(), чтобы пользователь мог видеть ошибку

    # --- Отображение Результатов ---
    if webrtc_ctx: # Только если стример успешно инициализирован
        col1, col2 = st.columns([3, 2]) # Колонки для видео/статуса и данных/графика

        with col1:
            st.subheader("Видеопоток с Камеры")
            # Видео отображается компонентом webrtc_streamer выше
            st.caption("На видео должен быть виден зеленый прямоугольник вокруг лица для измерения.")
            status_placeholder = st.empty() # Для сообщений о статусе

        with col2:
            st.subheader("Результаты")
            hr_placeholder = st.empty()
            fps_placeholder = st.empty()
            st.subheader("rPPG Сигнал (Отфильтрованный)")
            signal_placeholder = st.empty()
            st.caption("График показывает извлеченный сигнал пульса.")

        # --- Цикл Обновления Интерфейса ---
        # Обновляем UI регулярно, пока стрим активен
        while webrtc_ctx.state.playing and "rppg_processor" in st.session_state:
            try:
                # Получаем данные из процессора (потокобезопасно через геттеры)
                processor = st.session_state.rppg_processor
                current_status = processor.get_status()
                current_smoothed_hr = processor.get_hr()
                current_fps = processor.get_fps()
                plot_data = processor.get_plot_data()

                # Обновляем плейсхолдеры
                status_placeholder.info(f"Статус: {current_status}", icon="ℹ️")

                hr_display = f"{current_smoothed_hr:.1f}" if not np.isnan(current_smoothed_hr) else "---"
                hr_placeholder.metric("Сглаженный Пульс (уд/мин)", hr_display)

                fps_placeholder.metric("Обработка FPS", f"{current_fps:.1f}")

                # Обновляем график, если есть данные
                if plot_data is not None and plot_data.size > 10: # Требуем минимум 10 точек для графика
                     # Создаем DataFrame для лучшего отображения в line_chart
                     # chart_data = pd.DataFrame({'rPPG Signal': plot_data})
                     # signal_placeholder.line_chart(chart_data)
                     signal_placeholder.line_chart(plot_data) # Прямая передача NumPy массива
                else:
                     signal_placeholder.empty() # Очищаем график, если нет данных

                # Пауза для контроля частоты обновления UI (слишком частые обновления могут тормозить)
                time.sleep(0.15) # Обновляем UI примерно 6-7 раз в секунду

            except AttributeError as e_attr:
                 # Эта ошибка может возникнуть, если процессор был удален из session_state во время работы цикла
                 print(f"Ошибка доступа к rppg_processor в цикле UI: {e_attr}")
                 status_placeholder.error("Ошибка: Процессор был неожиданно удален. Пожалуйста, перезагрузите страницу.", icon="🆘")
                 break # Выходим из цикла обновления UI
            except Exception as e_ui:
                 # Ловим другие ошибки обновления UI, но не прерываем цикл
                 print(f"Ошибка в цикле обновления UI: {e_ui}")
                 # Можно добавить st.warning() для пользователя
                 # traceback.print_exc() # Для детальной отладки
                 time.sleep(0.5) # Пауза подольше при ошибке

        # --- Состояние после остановки стрима ---
        if not webrtc_ctx.state.playing:
             print(">>> WebRTC стрим остановлен.")
             # Очищаем плейсхолдеры или показываем сообщение о неактивности
             status_placeholder.warning("Камера не активна. Нажмите 'START' выше, чтобы начать.", icon="⚠️")
             hr_placeholder.metric("Сглаженный Пульс (уд/мин)", "N/A")
             fps_placeholder.metric("Обработка FPS", "N/A")
             signal_placeholder.empty()
             # Можно добавить кнопку для перезапуска стрима без перезагрузки страницы,
             # но это может быть сложно с текущей архитектурой streamlit-webrtc
             # if st.button("Попробовать снова"):
             #      st.rerun() # Простой способ - перезагрузить всю страницу

    else:
         # Если webrtc_ctx не был создан (ошибка при запуске стримера)
         st.warning("Не удалось инициализировать видеопоток.", icon="📹")


# --- Запуск Приложения ---
if __name__ == "__main__":
    print("="*50)
    print("Запуск Streamlit rPPG приложения...")
    print(f"Текущий рабочий каталог: {os.getcwd()}")
    print("Проверка наличия файлов моделей:")
    models_exist = False
    for name, path in NN_MODEL_PATHS.items():
         if path and os.path.exists(path):
              print(f"  [OK] {name}: {path}")
              models_exist = True
         elif path:
              print(f"  [!] {name}: Файл НЕ НАЙДЕН по пути {path}")
         else:
              print(f"  [?] {name}: Путь не указан.")

    if not models_exist:
         print("\nПРЕДУПРЕЖДЕНИЕ: Не найден ни один файл модели НС.")
         print("Доступны будут только классические методы (POS, CHROM, ICA).")
         print("Убедитесь, что файлы .pth находятся в нужном месте или указаны правильные пути в NN_MODEL_PATHS.")
    print("="*50)

    # Запускаем основную функцию Streamlit
    run_streamlit_app()