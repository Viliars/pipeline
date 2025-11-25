import os
import tempfile
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .efficientnet import EfficientNet, params
from .mel_utils import compute_and_save_mel_image

import warnings
warnings.filterwarnings("ignore")

class WavPredictor:
    """Класс для предсказания классов wav файлов через мел-спектрограммы"""
    def __init__(self, model_path: str, model_type: str, label_map: Dict[int, str], 
                 num_classes: int = None, device: str = None, duration_sec: float = 7.0):
        """
        Инициализация предсказателя
        
        Args:
            model_path: Путь к файлу модели (.pth)
            model_type: Тип модели (efficientnet_b0, efficientnet_b1, ..., efficientnet_b7)
            label_map: Словарь соответствия номера класса и имени {0: 'noise', 1: 'fire'}
            num_classes: Количество классов (определяется автоматически из label_map)
            device: Устройство для вычислений ('cuda', 'cpu', или None для автоопределения)
            duration_sec: Длительность аудио в секундах для мел-спектрограммы (по умолчанию 7.0)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.label_map = label_map
        self.num_classes = num_classes if num_classes else len(label_map)
        self.duration_sec = duration_sec
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Параметры модели
        if model_type not in params:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Доступные: {list(params.keys())}")
        
        self.width_mult, self.depth_mult, self.image_size, self.dropout_rate = params[model_type]
        
        # Загрузка модели
        self.model = self._load_model()
        
        # Трансформации для изображений (мел-спектрограммы будут обрабатываться как изображения)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self) -> nn.Module:
        # Создание модели
        model = EfficientNet(self.width_mult, self.depth_mult, self.dropout_rate)
        
        # Загрузка checkpoint'а
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Извлечение state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Обработка state_dict (удаление 'module.' префикса если есть)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # удаляем 'module.'
            new_state_dict[k] = v
        
        # Определение количества каналов для классификатора
        base_channels = 1280
        last_channels = int(base_channels * self.width_mult)
        
        # Специфичные значения для некоторых моделей
        if self.model_type == 'efficientnet_b6':
            last_channels = 2304
        elif self.model_type == 'efficientnet_b7':
            last_channels = 2560
        elif self.model_type == 'efficientnet_b5':
            last_channels = 2048
        
        # Создание классификатора
        model.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(last_channels, self.num_classes),
            nn.Softmax(dim=1)  # Добавляем Softmax для получения вероятностей
        )
        
        # Загрузка весов
        model_dict = model.state_dict()
        new_state_dict_filtered = {k: v for k, v in new_state_dict.items() 
                                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(new_state_dict_filtered)
        model.load_state_dict(model_dict)

        model.to(self.device)
        model.eval()

        # print(f"✅ Модель загружена: {self.model_type}")
        # print(f"   Размер изображения: {self.image_size}x{self.image_size}")
        # print(f"   Количество классов: {self.num_classes}")
        # print(f"   Классы: {self.label_map}")
        
        return model

    def _wav_to_mel_image(self, wav_path: str, output_image_path: str, 
                          sr: int = 16000, n_fft: int = 400, 
                          hop_length: int = 100, win_length: int = 400,
                          n_mels: int = 80, duration_sec: float = 7.0,
                          width: int = 640, height: int = 480) -> None:
        """
        Конвертация wav файла в мел-спектрограмму (изображение)
        
        Args:
            wav_path: Путь к wav файлу
            output_image_path: Путь для сохранения изображения мел-спектрограммы
            sr: Частота дискретизации (по умолчанию 16000 как в CMGAN)
            n_fft: Размер FFT (по умолчанию 400 как в CMGAN)
            hop_length: Шаг окна (по умолчанию 100 как в CMGAN)
            win_length: Длина окна (по умолчанию 400 как в CMGAN)
            n_mels: Число мел-полос (по умолчанию 80 как в CMGAN)
            duration_sec: Длительность аудио в секундах (по умолчанию 7.0)
            width: Ширина выходного изображения
            height: Высота выходного изображения
        """
        compute_and_save_mel_image(
            wav_path=wav_path,
            out_image_path=output_image_path,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=0.0,
            fmax=None,
            power=2.0,
            to_db=True,
            cmap='magma',
            normalize=True,
            target_size=(width, height),
            duration_sec=duration_sec,
        )

    def predict_single_wav(self, wav_path: str, temp_dir: str = None) -> Tuple[int, str, torch.Tensor]:
        """
        Предсказание класса для одного wav файла
        
        Args:
            wav_path: Путь к wav файлу
            temp_dir: Директория для временных файлов (если None, используется системная)
        
        Returns:
            predicted_class: Номер предсказанного класса
            class_name: Имя предсказанного класса
            probabilities: Вероятности для всех классов
        """
        try:
            # Создаем временное изображение мел-спектрограммы
            if temp_dir is None:
                temp_dir = tempfile.gettempdir()
            
            temp_image_path = os.path.join(temp_dir, f"temp_mel_{os.path.basename(wav_path)}.png")
            
            # Конвертируем wav в мел-спектрограмму
            self._wav_to_mel_image(wav_path, temp_image_path, duration_sec=self.duration_sec)
            
            # Загрузка и преобразование изображения мел-спектрограммы
            image = Image.open(temp_image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = outputs[0]  # Уже после Softmax
                predicted_class = torch.argmax(probabilities).item()

            # Получение имени класса
            class_name = self.label_map.get(predicted_class, f"unknown_{predicted_class}")
            
            # Удаляем временный файл
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            return predicted_class, class_name, probabilities
            
        except Exception as e:
            print(f"⚠️  Ошибка при обработке {wav_path}: {e}")
            return -1, "error", None
        
label_map = {
    0: "Clean",
    1: "3A",
    2: "Additive",
    3: "Artefacts",
    4: "Babble",
    5: "Reverb",
    6: "Combined"
}

model_path = "models/predictor.ckpt"

predictor = WavPredictor(
    model_path=model_path,
    model_type="efficientnet_b0_lite",
    label_map=label_map,
    duration_sec=7.0
)

def predict_class(wav_file: str) -> Tuple[int, str]:
    class_number, class_name, _ = predictor.predict_single_wav(wav_file)

    return class_number, class_name
