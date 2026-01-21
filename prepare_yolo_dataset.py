import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import warnings
import os
import json
import random
import shutil
from pathlib import Path

from collections import defaultdict
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# Подключаем датасет
project_dir = './data/segmentation'

video_dir = project_dir + "/videos"
masks_dir = project_dir + "/masks"

videos_file = video_dir +  '/ID100_ARTERIAL.npy'
masks_file = masks_dir + '/ID100_ARTERIAL_MASK.npy'

try:
    videos = np.load(videos_file)
    print("Файл 'videos' загружен успешно.")
except FileNotFoundError:
    print("Файл 'videos' не найден. Проверьте правильность пути.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path("./")  # рабочая папка (можно поменять)
DATASET_ROOT = Path("./data/segmentation")  #  videos/, masks/, format.json
FORMAT_JSON = DATASET_ROOT / "format.json"

OUTPUT_BASE = ROOT / "data/data"  # train/val и data.yaml
PATHS_VIDEOMASKS_JSON = ROOT / "data/pathsVideoMasks.json"
RESERVE_VIDEOS_JSON = ROOT / "data/reserved_videos.json"  # 20% тест
USE_VIDEO_FRACTION = 0.8  # 80% видео для аналитики/обучения
VAL_RATIO_IN_SELECTED = 0.1  #10% валидации от выбранных кадров

IMG_SHAPE = (224, 224)
CLASS_NAME = "tumor"
CLASS_INDEX = 0

os.makedirs(ROOT, exist_ok=True)
print("CONFIG:", {"ROOT": str(ROOT), "DATASET_ROOT": str(DATASET_ROOT), "FORMAT_JSON": str(FORMAT_JSON)})

# Cell 2 - собрать словарь video -> [masks] и сохранить
with open(FORMAT_JSON, "r", encoding="utf-8") as f:
    format_data = json.load(f)

records = list(format_data.values()) if isinstance(format_data, dict) else list(format_data)
paths_vm = defaultdict(list)

def resolve_to_dataset_root(p_raw: str, dataset_root: Path) -> str:
    """
    Приводит путь из format.json к реальному пути в файловой системе:
    - если путь начинается с 'data/segmentation' -> заменяет этот префикс на DATASET_ROOT
    - если указанный путь существует как есть -> возвращает его
    - если не существует -> пробует dataset_root / p_raw
    - если всё ещё не найдено -> пробует Path.cwd() / p_raw
    Возвращает строку пути (необязательно resolved), сохраняет как абсолютный путь, если найден.
    """
    if p_raw is None:
        return None
    p = str(p_raw)

    # 1) специальная замена префикса 'data/segmentation' -> DATASET_ROOT
    if p.startswith("data/segmentation"):
        replaced = p.replace("data/segmentation", str(dataset_root))
        cand = Path(replaced)
        if cand.exists():
            return str(cand)

    # 2) если путь уже существует как есть
    cand = Path(p)
    if cand.exists():
        return str(cand)

    # 3) попробовать как относительный внутри DATASET_ROOT
    cand = dataset_root / cand
    if cand.exists():
        return str(cand)

    # 4) попробовать относительный к текущей рабочей директории
    cand = Path.cwd() / p
    if cand.exists():
        return str(cand)

    # 5) вернуть best-effort (DATASET_ROOT / исходное имя файла) — даже если не существует, это то что хотели
    #    тут мы пытаемся сохранить ожидаемый абсолютный путь под DATASET_ROOT
    fallback = dataset_root / Path(p).name
    return str(fallback)

for rec in records:
    vpath = rec.get("video_path")
    mpath = rec.get("mask_path")
    if vpath is None or mpath is None:
        continue

    v_res = resolve_to_dataset_root(vpath, DATASET_ROOT)
    m_res = resolve_to_dataset_root(mpath, DATASET_ROOT)

    paths_vm[v_res].append(m_res)

# Сохранение словаря
with open(PATHS_VIDEOMASKS_JSON, "w", encoding="utf-8") as fout:
    json.dump(paths_vm, fout, ensure_ascii=False, indent=2)

print(f"Собрано видео: {len(paths_vm)}. Пример (пара первых):")
for i, (k, v) in enumerate(paths_vm.items()):
    print(i+1, k, "->", len(v), "masks")
    if i >= 4: break

# Cell 3 - разделение видео: используем 80%, оставляем 20% для независимого теста
all_videos = list(paths_vm.keys())
all_videos.sort()
n_all = len(all_videos)
n_use = int(round(USE_VIDEO_FRACTION * n_all))
random.Random(SEED).shuffle(all_videos)

selected_videos = all_videos[:n_use]
reserved_videos = all_videos[n_use:]

with open(RESERVE_VIDEOS_JSON, "w", encoding="utf-8") as f:
    json.dump({"reserved": reserved_videos, "selected": selected_videos}, f, ensure_ascii=False, indent=2)

print(f"Всего видео: {n_all}; используем: {len(selected_videos)}; резерв: {len(reserved_videos)}")
print("Первые 3 выбранных:", selected_videos[:3])
print("Первые 3 зарезервированных:", reserved_videos[:3])

# Cell 4 - пройти видео по одному, без загрузки всего в память
positive_frames = []  # (video_path, frame_index)
negative_frames = []  # (video_path, frame_index)
video_frame_counts = {}  # для отладки: сколько кадров в видео

def ensure_bin_mask(arr):
    """Привести маску к бинарной 0/255"""
    # arr shape (T,H,W)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    # считаем >127 как белое
    bin_arr = np.where(arr > 127, 255, 0).astype(np.uint8)
    return bin_arr

for vp in selected_videos:
    vp_path = Path(vp)
    if not vp_path.exists():
        # попробуем DATASET_ROOT / vp
        testp = Path.cwd() / vp
        if testp.exists():
            vp_path = testp
        else:
            alt = DATASET_ROOT / vp_path.name
            if alt.exists():
                vp_path = alt
            else:
                print("WARNING: видео не найдено:", vp)
                continue

    # Загружаем видео (shape (T,H,W))
    video = np.load(vp_path)
    T = video.shape[0]
    video_frame_counts[vp] = T

    # Загружаем все маски, связанные с этим видео (каждая маска shape (T,H,W))
    mask_paths = paths_vm[vp]
    masks_list = []
    for mp in mask_paths:
        mp_path = Path(mp)
        if not mp_path.exists():
            mp_try = Path.cwd() / mp
            if mp_try.exists():
                mp_path = mp_try
            else:
                alt = DATASET_ROOT / mp_path.name
                if alt.exists():
                    mp_path = alt
                else:
                    print("WARNING: маска не найдена:", mp)
                    continue
        m = np.load(mp_path)
        masks_list.append(ensure_bin_mask(m))

    if len(masks_list) == 0:
        # нет масок вообще для этого видео -> все кадры негативные
        for t in range(T):
            negative_frames.append((vp, t))
        continue

    # Объединяем логически кадр по кадру (OR) при проверке наличия пикселей:
    # Не создаём гигантский массив, обрабатываем кадры по одному
    for t in range(T):
        # проверяем, есть ли ненулевые пиксели в любом mask[:,t,:,:]
        has = False
        for m in masks_list:
            if np.any(m[t] != 0):
                has = True
                break
        if has:
            positive_frames.append((vp, t))
        else:
            negative_frames.append((vp, t))

print("Итого кадров (positive, negative):", len(positive_frames), len(negative_frames))
# суммарное число кадров видео из датасета (из всех видео) для которых хотя бы на одной маске есть выделенный пиксель:
total_positive = len(positive_frames)
print("Суммарное количество кадров с выделениями:", total_positive)

if total_positive == 0:
    raise RuntimeError("Ни одного положительного кадра не найдено — проверьте маски!")

if len(negative_frames) < total_positive:
    print("WARNING: недостаточно негативных кадров, используем все и продублируем (редко).")
    # на всякий случай — репит негативов (лучше собрать больше данных)
    multiplier = int(np.ceil(total_positive / max(1, len(negative_frames))))
    negative_frames_expanded = negative_frames * multiplier
    negative_selected = negative_frames_expanded[:total_positive]
else:
    negative_selected = random.Random(SEED).sample(negative_frames, total_positive)

print("Выбрано негативных кадров:", len(negative_selected), "для баланса с позитивными:", total_positive)

# объединяем и перемешиваем
combined = [(vp, t, 1) for vp, t in positive_frames] + [(vp, t, 0) for vp, t in negative_selected]
random.Random(SEED).shuffle(combined)
print("Итого примеров (после балансировки):", len(combined))

train_examples, val_examples = train_test_split(combined, test_size=VAL_RATIO_IN_SELECTED, random_state=SEED, shuffle=True)

print("Train examples:", len(train_examples), "Val examples:", len(val_examples))

# Подготовка директорий
def prepare_dirs(base_out):
    img_dir = base_out / "images"
    lbl_dir = base_out / "labels"
    if base_out.exists():
        # не удаляем полностью (можно раскомментировать удаление при желании)
        pass
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir

train_img_dir, train_lbl_dir = prepare_dirs(OUTPUT_BASE / "train")
val_img_dir, val_lbl_dir = prepare_dirs(OUTPUT_BASE / "val")

# Счетчик для уникальных имён файлов
global_counter = 0

def save_example(vp, t, positive, out_img_dir, out_lbl_dir, counter):
    """Сохраняет кадр t из видео vp как counter.png и соответствующий .txt.
       Для положительного кадра строит контуры по объединённой маске (все маски OR).
    """
    # загружаем видео и все маски для данного видео (можно кешировать, но для простоты читаем)
    vp_path = Path(vp)
    if not vp_path.exists():
        vp_path = Path.cwd() / vp_path
    video = np.load(vp_path)  # (T,H,W)
    frame = video[t].astype(np.uint8)
    # сохраняем изображение как BGR
    fn = f"{counter:06d}.png"
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_img_dir / fn), bgr)

    # собираем объединённую маску для этого кадра
    masks = []
    for mp in paths_vm[vp]:
        mp_path = Path(mp)
        if not mp_path.exists():
            mp_path = Path.cwd() / mp_path
        if mp_path.exists():
            m = np.load(mp_path)
            masks.append(ensure_bin_mask(m)[t])
    if len(masks) == 0:
        combined_mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
    else:
        # OR всех масок
        combined_mask = np.zeros_like(masks[0])
        for mm in masks:
            combined_mask = np.logical_or(combined_mask, mm).astype(np.uint8) * 255

    # теперь ищем контуры и создаём .txt (инстанс-полигоны)
    txt_path = out_lbl_dir / fn.replace(".png", ".txt")
    lines = []
    if positive:
        # найдем контуры (в OpenCV нужно бинарное изображение uint8)
        ms = combined_mask.copy()
        if ms.max() > 1:
            ms = (ms > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = ms.shape
        for cnt in contours:
            if cnt is None:
                continue
            cnt = cnt.squeeze()
            if cnt.ndim != 2 or cnt.shape[0] < 3:
                continue
            # Полигон точек (x,y) flatten
            poly = cnt.flatten().tolist()
            # Нормализация
            norm = []
            for idx_c, coord in enumerate(poly):
                if idx_c % 2 == 0:  # x
                    norm.append(coord / W)
                else:               # y
                    norm.append(coord / H)
            # YOLOv8-seg: class-index + polygon coords
            lines.append(str(CLASS_INDEX) + " " + " ".join(f"{v:.6f}" for v in norm))
        # если не нашлось контуров (редкость) — записываем пустой файл
    # Сохраняем txt (может быть пустой)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return fn

# Сохраняем train
print("Экспорт train...")
for ex in train_examples:
    vp, t, pos = ex
    fn = save_example(vp, t, pos, train_img_dir, train_lbl_dir, global_counter)
    global_counter += 1
    if global_counter % 500 == 0:
        print("Сохранено примеров:", global_counter)

# Сохраняем val
print("Экспорт val...")
for ex in val_examples:
    vp, t, pos = ex
    fn = save_example(vp, t, pos, val_img_dir, val_lbl_dir, global_counter)
    global_counter += 1

print("Готово. Всего сохранено файлов:", global_counter)

import yaml

yaml_config = {
    'path': str(OUTPUT_BASE),  # относительный путь к data
    'train': 'train/images',
    'val': 'val/images',
    'names': {CLASS_INDEX: CLASS_NAME}
}

with open(OUTPUT_BASE / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(yaml_config, f, sort_keys=False, allow_unicode=True)

print("data.yaml сохранён по пути:", OUTPUT_BASE / "data.yaml")
print("Краткий отчёт:")
print("  Положительных кадров (с сегментацией) использовано:", total_positive)
print("  Негативных (баланс) использовано:", len(negative_selected))
print("  Итого сохранённых кадров:", len(combined))
print("  Train:", len(train_examples), " Val:", len(val_examples))
print("  Папки с изображениями (пример):", train_img_dir, val_img_dir)
print("  Словарь pathsVideoMasks сохранён:", PATHS_VIDEOMASKS_JSON)
print("  Резервные видео (20%) сохранены:", RESERVE_VIDEOS_JSON)