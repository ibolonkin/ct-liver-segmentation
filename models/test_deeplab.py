# ====================
# 📦 ИМПОРТЫ И НАСТРОЙКИ
# ====================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pydicom
from tqdm import tqdm
import glob
import json
import segmentation_models_pytorch as smp

# ====================
# 🧠 МОДЕЛЬ DeepLabV3+ (ИСПРАВЛЕННАЯ ДЛЯ ЗАГРУЗКИ КЛЮЧЕЙ)
# ====================

class DeepLabV3PlusModel:
    def __init__(self, num_classes=3, encoder_name='resnet50', encoder_weights='imagenet'):
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
            in_channels=1  # 1 канал для CT снимков
        )

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

# ====================
# 🔧 ЭТАП 1: ЗАГРУЗКА МОДЕЛИ DEEPLABV3+ (ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ)
# ====================

def load_trained_deeplab_model(model_path, device='cuda'):
    """
    Загрузка обученной модели DeepLabV3+ с исправлением ключей
    """
    print(f"📥 Загрузка DeepLabV3+ модели из: {model_path}")

    try:
        # 1. Создаем модель
        model = DeepLabV3PlusModel(
            num_classes=3,
            encoder_name='resnet50',
            encoder_weights=None
        )

        # 2. Загружаем веса из checkpoint
        print(f"🔍 Загрузка checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)

        # 3. Проверяем тип checkpoint
        if isinstance(checkpoint, dict):
            print(f"✅ Checkpoint является словарем (state_dict)")
            print(f"📊 Ключей в checkpoint: {len(checkpoint)}")

            # 4. ПЕРЕИМЕНОВЫВАЕМ КЛЮЧИ для совместимости
            print("🔄 Переименование ключей для совместимости...")
            new_checkpoint = {}

            for key, value in checkpoint.items():
                # Преобразуем ключи из формата model.backbone.* в encoder.*
                if key.startswith('model.backbone.'):
                    # Заменяем model.backbone. на encoder.
                    new_key = key.replace('model.backbone.', 'encoder.')
                    new_checkpoint[new_key] = value
                # Преобразуем ключи из формата model.classifier.* в decoder.*
                elif key.startswith('model.classifier.'):
                    # Преобразуем сложную структуру
                    if 'convs.0.0' in key:
                        new_key = key.replace('model.classifier.0.convs.0.0', 'decoder.aspp.0.convs.0.0')
                    elif 'convs.0.1' in key:
                        new_key = key.replace('model.classifier.0.convs.0.1', 'decoder.aspp.0.convs.0.1')
                    elif 'convs.1.0' in key:
                        new_key = key.replace('model.classifier.0.convs.1.0', 'decoder.aspp.0.convs.1.0.0')
                    elif 'convs.1.1' in key:
                        new_key = key.replace('model.classifier.0.convs.1.1', 'decoder.aspp.0.convs.1.1')
                    elif 'convs.2.0' in key:
                        new_key = key.replace('model.classifier.0.convs.2.0', 'decoder.aspp.0.convs.2.0.0')
                    elif 'convs.2.1' in key:
                        new_key = key.replace('model.classifier.0.convs.2.1', 'decoder.aspp.0.convs.2.1')
                    elif 'convs.3.0' in key:
                        new_key = key.replace('model.classifier.0.convs.3.0', 'decoder.aspp.0.convs.3.0.0')
                    elif 'convs.3.1' in key:
                        new_key = key.replace('model.classifier.0.convs.3.1', 'decoder.aspp.0.convs.3.1')
                    elif 'convs.4.1' in key:
                        new_key = key.replace('model.classifier.0.convs.4.1', 'decoder.aspp.0.convs.4.1')
                    elif 'convs.4.2' in key:
                        new_key = key.replace('model.classifier.0.convs.4.2', 'decoder.aspp.0.convs.4.2')
                    elif 'project.0' in key:
                        new_key = key.replace('model.classifier.0.project.0', 'decoder.aspp.0.project.0')
                    elif 'project.1' in key:
                        new_key = key.replace('model.classifier.0.project.1', 'decoder.aspp.0.project.1')
                    elif key == 'model.classifier.1.weight':
                        new_key = 'decoder.aspp.1.0.weight'
                    elif key == 'model.classifier.2.weight':
                        new_key = 'decoder.aspp.1.1.weight'
                    elif key == 'model.classifier.2.bias':
                        new_key = 'decoder.aspp.2.weight'
                    elif key == 'model.classifier.2.running_mean':
                        new_key = 'decoder.aspp.2.bias'
                    elif key == 'model.classifier.2.running_var':
                        new_key = 'decoder.aspp.2.running_mean'
                    elif key == 'model.classifier.2.num_batches_tracked':
                        new_key = 'decoder.aspp.2.running_var'
                    elif key == 'model.classifier.4.weight':
                        new_key = 'decoder.block1.0.weight'
                    elif key == 'model.classifier.4.bias':
                        new_key = 'decoder.block1.1.weight'
                    else:
                        new_key = key
                    new_checkpoint[new_key] = value
                # Игнорируем aux_classifier если он есть
                elif key.startswith('model.aux_classifier.'):
                    continue
                else:
                    new_checkpoint[key] = value

            # 5. Загружаем исправленные веса
            print("🔄 Загрузка переименованных весов...")
            model.model.load_state_dict(new_checkpoint, strict=False)

            # 6. Проверяем, какие ключи загрузились
            model_keys = set(model.model.state_dict().keys())
            checkpoint_keys = set(new_checkpoint.keys())

            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys

            print(f"✅ Модель DeepLabV3+ успешно загружена!")
            print(f"📊 Статус загрузки:")
            print(f"   Отсутствующие ключи: {len(missing_keys)}")
            print(f"   Неожиданные ключи: {len(unexpected_keys)}")

            if len(missing_keys) > 0:
                print(f"   Пример отсутствующих ключей:")
                for i, key in enumerate(list(missing_keys)[:3]):
                    print(f"     - {key}")

        else:
            print(f"❌ Неизвестный тип checkpoint: {type(checkpoint)}")
            return None

        # 7. Переводим модель на устройство и в режим eval
        model.to(device)
        model.eval()

        # 8. Проверяем параметры модели
        total_params = sum(p.numel() for p in model.model.parameters())

        print(f"📊 Параметры модели:")
        print(f"   Всего параметров: {total_params:,}")
        print(f"   Устройство: {device}")

        # 9. Быстрая проверка работы модели
        print(f"🧪 Проверка работы модели...")
        try:
            test_input = torch.randn(1, 1, 256, 256).to(device)
            with torch.no_grad():
                test_output = model.model(test_input)
            print(f"✅ Тест пройден! Выходной размер: {test_output.shape}")
            print(f"   Ожидалось: [1, 3, 256, 256] для 3 классов")
        except Exception as e:
            print(f"⚠️  Предупреждение при тесте модели: {e}")

        return model

    except Exception as e:
        print(f"❌ Ошибка загрузки модели DeepLabV3+: {e}")
        import traceback
        traceback.print_exc()
        return None

# ====================
# 🔧 ЭТАП 2: ПОИСК DICOM ФАЙЛОВ
# ====================

def find_dicom_files(root_path):
    print(f"🔍 Поиск DICOM файлов в: {root_path}")

    dicom_extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']
    dicom_files = []

    for extension in dicom_extensions:
        pattern = os.path.join(root_path, '**', extension)
        files = glob.glob(pattern, recursive=True)
        dicom_files.extend(files)

    print(f"📁 Найдено {len(dicom_files)} DICOM файлов")

    if len(dicom_files) == 0:
        print("❌ DICOM файлы не найдены! Проверьте путь и расширения файлов.")

    return dicom_files

# ====================
# 🔧 ЭТАП 3: ПРЕДОБРАБОТКА DICOM ДЛЯ DEEPLABV3+
# ====================

def preprocess_dicom_for_deeplab(dicom_path, target_size=256):
    """
    Предобработка DICOM файлов специально для DeepLabV3+
    """
    try:
        # Чтение DICOM файла
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array

        # Применение rescale slope и intercept если доступно
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            pixel_array = pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept

        # Нормализация [0, 1] как в нашем пайплайне
        pixel_normalized = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)

        # Конвертация в uint8 [0, 255] для сохранения
        pixel_uint8 = (pixel_normalized * 255).astype(np.uint8)

        # Изменение размера до target_size
        pixel_resized = cv2.resize(pixel_uint8, (target_size, target_size))

        # Окончательная нормализация [0, 1] для модели
        pixel_float = pixel_resized.astype(np.float32) / 255.0

        # Создание тензора для DeepLabV3+ (добавляем канальное измерение)
        tensor = torch.FloatTensor(pixel_float).unsqueeze(0)  # [1, H, W] для DeepLabV3+

        # Метаданные
        metadata = {
            'original_shape': pixel_array.shape,
            'filename': os.path.basename(dicom_path),
            'patient_id': getattr(dicom, 'PatientID', 'Unknown'),
            'study_date': getattr(dicom, 'StudyDate', 'Unknown'),
            'processed_shape': pixel_resized.shape
        }

        return tensor, pixel_array, metadata

    except Exception as e:
        print(f"❌ Ошибка обработки {dicom_path}: {e}")
        return None, None, None

# ====================
# 🔧 ЭТАП 4: ИНФЕРЕНС МОДЕЛИ DEEPLABV3+
# ====================

def predict_single_slice_deeplab(model, dicom_tensor, device='cuda'):
    """
    Предсказание для одного среза с DeepLabV3+
    """
    with torch.no_grad():
        dicom_tensor = dicom_tensor.to(device)
        output = model.model(dicom_tensor.unsqueeze(0))  # Добавляем batch dimension
        prediction = torch.argmax(output, dim=1)
        return prediction.squeeze().cpu().numpy()

# ====================
# 🔧 ЭТАП 5: УПРОЩЕННАЯ ВИЗУАЛИЗАЦИЯ (ДЛЯ БОЛЬШОГО КОЛИЧЕСТВА ФАЙЛОВ)
# ====================

def quick_visualization(original, prediction, filename="", save_path=None):
    """
    Быстрая визуализация для большого количества файлов
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Оригинальное изображение
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original: {filename[:30]}...')
    axes[0].axis('off')

    # Предсказание
    axes[1].imshow(prediction, cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ====================
# 🔧 ЭТАП 6: АНАЛИЗ СТАТИСТИКИ ПРЕДСКАЗАНИЙ
# ====================

def analyze_prediction_statistics(prediction):
    """
    Анализирует статистику предсказания
    """
    total_pixels = prediction.size

    # Подсчет пикселей по классам
    background_pixels = np.sum(prediction == 0)
    liver_pixels = np.sum(prediction == 1)
    tumor_pixels = np.sum(prediction == 2)

    # Проценты
    background_percentage = background_pixels / total_pixels
    liver_percentage = liver_pixels / total_pixels
    tumor_percentage = tumor_pixels / total_pixels

    # Флаги обнаружения
    liver_detected = liver_pixels > 0
    tumor_detected = tumor_pixels > 0

    return {
        'background_pixels': background_pixels,
        'liver_pixels': liver_pixels,
        'tumor_pixels': tumor_pixels,
        'background_percentage': background_percentage,
        'liver_percentage': liver_percentage,
        'tumor_percentage': tumor_percentage,
        'liver_detected': liver_detected,
        'tumor_detected': tumor_detected,
        'total_pixels': total_pixels
    }

# ====================
# 🔧 ЭТАП 7: ОСНОВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ (ОПТИМИЗИРОВАННАЯ)
# ====================

def test_deeplab_on_dicom_files(model_path, dicom_root_path, output_dir='deeplab_dicom_test_results',
                              target_size=256, device='cuda', max_samples_to_visualize=20):
    """
    Оптимизированная функция для тестирования на большом количестве файлов
    """
    # Создаем папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sample_visualizations'), exist_ok=True)

    print("🚀 НАЧАЛО ТЕСТИРОВАНИЯ DICOM С DeepLabV3+")
    print("=" * 60)

    # ЭТАП 1: Загрузка модели DeepLabV3+
    model = load_trained_deeplab_model(model_path, device)
    if model is None:
        return None

    # ЭТАП 2: Поиск DICOM файлов
    dicom_files = find_dicom_files(dicom_root_path)
    if len(dicom_files) == 0:
        return None

    # ЭТАП 3: Обработка файлов с интеллектуальной выборкой для визуализации
    print(f"\n🔬 Обработка {len(dicom_files)} DICOM файлов с DeepLabV3+...")

    results = []
    processed_count = 0
    visualization_count = 0

    # Определяем шаг для визуализации (чтобы показать max_samples_to_visualize файлов)
    if len(dicom_files) > max_samples_to_visualize:
        visualize_step = len(dicom_files) // max_samples_to_visualize
    else:
        visualize_step = 1

    # Прогресс-бар
    progress_bar = tqdm(dicom_files, desc="Обработка DICOM файлов", unit="файл")

    for i, dicom_path in enumerate(progress_bar):
        try:
            # Предобработка DICOM
            tensor, original_array, metadata = preprocess_dicom_for_deeplab(dicom_path, target_size)

            if tensor is not None:
                # Предсказание
                prediction = predict_single_slice_deeplab(model, tensor, device)

                # Анализ статистики
                stats = analyze_prediction_statistics(prediction)

                # Сохраняем результат
                result = {
                    'file_path': dicom_path,
                    'filename': metadata['filename'],
                    'patient_id': metadata['patient_id'],
                    'background_percentage': stats['background_percentage'],
                    'liver_percentage': stats['liver_percentage'],
                    'tumor_percentage': stats['tumor_percentage'],
                    'liver_detected': stats['liver_detected'],
                    'tumor_detected': stats['tumor_detected'],
                    'liver_pixels': stats['liver_pixels'],
                    'tumor_pixels': stats['tumor_pixels']
                }

                results.append(result)
                processed_count += 1

                # УСЛОВНАЯ ВИЗУАЛИЗАЦИЯ: только для выборочных файлов
                should_visualize = (
                    visualization_count < max_samples_to_visualize and
                    (i % visualize_step == 0 or stats['tumor_detected'] or stats['liver_pixels'] > 1000)
                )

                if should_visualize:
                    visualization_count += 1
                    original_resized = cv2.resize(original_array, (target_size, target_size))

                    # Сохраняем визуализацию в файл вместо показа
                    save_path = os.path.join(output_dir, 'sample_visualizations',
                                           f"sample_{visualization_count:03d}_{metadata['filename'][:20]}.png")

                    quick_visualization(
                        original_resized,
                        prediction,
                        metadata['filename'],
                        save_path
                    )

                    # Показываем только первые 5 визуализаций
                    if visualization_count <= 5:
                        print(f"\n📊 Пример {visualization_count}: {metadata['filename']}")
                        print(f"   Печень: {stats['liver_percentage']*100:.1f}%, Опухоль: {stats['tumor_percentage']*100:.1f}%")
                        img = Image.open(save_path)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(f"Пример {visualization_count}: {metadata['filename'][:30]}...")
                        plt.show()
                        plt.close()

                # Обновляем прогресс-бар
                if i % 100 == 0:
                    progress_bar.set_postfix({
                        'обработано': processed_count,
                        'печень': f"{sum(r['liver_detected'] for r in results)}/{len(results)}",
                        'опухоль': f"{sum(r['tumor_detected'] for r in results)}/{len(results)}"
                    })

        except Exception as e:
            progress_bar.write(f"❌ Ошибка в {dicom_path}: {str(e)[:100]}...")
            continue

    # ЭТАП 4: Расчет статистики
    print(f"\n📈 РАСЧЕТ СТАТИСТИКИ...")

    if len(results) > 0:
        results_df = pd.DataFrame(results)

        # Сохраняем детальные результаты
        results_csv_path = os.path.join(output_dir, 'deeplab_detailed_results.csv')
        results_df.to_csv(results_csv_path, index=False)

        # Статистика
        summary_stats = {
            'total_files': len(results_df),
            'files_with_liver': int(results_df['liver_detected'].sum()),
            'files_with_tumor': int(results_df['tumor_detected'].sum()),
            'avg_liver_percentage': float(results_df['liver_percentage'].mean() * 100),
            'avg_tumor_percentage': float(results_df['tumor_percentage'].mean() * 100),
            'max_liver_percentage': float(results_df['liver_percentage'].max() * 100),
            'max_tumor_percentage': float(results_df['tumor_percentage'].max() * 100),
            'avg_liver_pixels': float(results_df['liver_pixels'].mean()),
            'avg_tumor_pixels': float(results_df['tumor_pixels'].mean()),
            'visualization_samples': visualization_count
        }

        # Сохраняем статистику
        stats_json_path = os.path.join(output_dir, 'summary_statistics.json')
        with open(stats_json_path, 'w') as f:
            json.dump(summary_stats, f, indent=4)

        # Вывод сводки
        print("\n" + "=" * 60)
        print("📊 СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 60)
        print(f"📁 Всего обработано файлов: {summary_stats['total_files']}")
        print(f"🟢 Файлов с печенью: {summary_stats['files_with_liver']} ({summary_stats['files_with_liver']/summary_stats['total_files']*100:.1f}%)")
        print(f"🔴 Файлов с опухолью: {summary_stats['files_with_tumor']} ({summary_stats['files_with_tumor']/summary_stats['total_files']*100:.1f}%)")
        print(f"📈 Средний процент печени: {summary_stats['avg_liver_percentage']:.2f}%")
        print(f"📈 Средний процент опухоли: {summary_stats['avg_tumor_percentage']:.2f}%")
        print(f"👁️  Визуализировано примеров: {summary_stats['visualization_samples']}")
        print("=" * 60)

        # Быстрая визуализация распределений
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Распределение процентов печени
        axes[0].hist(results_df['liver_percentage'] * 100, bins=50, alpha=0.7, color='green')
        axes[0].set_xlabel('Процент печени (%)')
        axes[0].set_ylabel('Количество файлов')
        axes[0].set_title('Распределение процента печени')
        axes[0].grid(True, alpha=0.3)

        # Распределение процентов опухоли
        axes[1].hist(results_df['tumor_percentage'] * 100, bins=50, alpha=0.7, color='red')
        axes[1].set_xlabel('Процент опухоли (%)')
        axes[1].set_ylabel('Количество файлов')
        axes[1].set_title('Распределение процента опухоли')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distributions.png'), dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n💾 Результаты сохранены в: {output_dir}")
        print(f"   📄 Детальные результаты: {results_csv_path}")
        print(f"   📊 Сводная статистика: {stats_json_path}")
        print(f"   🖼️  Примеры визуализаций: {os.path.join(output_dir, 'sample_visualizations')}")

        return {
            'model': model,
            'results_df': results_df,
            'summary_stats': summary_stats
        }

    return None

# ====================
# 🚀 ЗАПУСК ТЕСТИРОВАНИЯ
# ====================

def main():
    """
    Основная функция запуска
    """
    # Укажите ваши пути здесь
    DEEPLAB_MODEL_PATH = "/content/drive/MyDrive/best_deeplabv3_liver_model.pth"
    DICOM_ROOT_PATH = "/content/drive/MyDrive/Anon_Liver/"

    print("🎯 НАЧАЛО ТЕСТИРОВАНИЯ DeepLabV3+")
    print(f"Модель: {DEEPLAB_MODEL_PATH}")
    print(f"DICOM файлы: {DICOM_ROOT_PATH}")
    print("=" * 60)

    # Запуск тестирования
    results = test_deeplab_on_dicom_files(
        model_path=DEEPLAB_MODEL_PATH,
        dicom_root_path=DICOM_ROOT_PATH,
        output_dir='deeplab_test_results',
        target_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_samples_to_visualize=20  # Показать только 20 примеров из 10621
    )

    if results:
        print("\n✅ ТЕСТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print(f"📊 Обработано: {results['summary_stats']['total_files']} файлов")
    else:
        print("\n❌ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО С ОШИБКАМИ")

if __name__ == "__main__":
    main()