import os
import pandas as pd
import requests
from urllib.parse import urlencode
import shutil
from tqdm import tqdm
import csv
import json
import cv2
import numpy as np
# from download_test import download_file_from_csv

def get_resource_info(public_link):
    """
    Получение информации о ресурсах из API Яндекс.Диска.
    """
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources?'
    final_url = base_url + urlencode(dict(public_key=public_link))
    response = requests.get(final_url)

    # Проверка успешности запроса
    if response.status_code != 200:
        raise ValueError(f"Ошибка получения данных: {response.status_code} - {response.text}")

    return response.json()

def extract_links_from_excel(input_excel, sheet_name):
    """
    Извлечение данных из Excel файла.
    """
    try:
        data = pd.read_excel(input_excel, sheet_name=sheet_name)
        # Удаление дубликатов ссылок на Яндекс.Диск
        data = data.drop_duplicates(subset=['Местоположение файлов'])
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{input_excel}' не найден.")
    except Exception as e:
        raise Exception(f"Произошла ошибка: {e}")

def read_existing_csv(output_csv):
    """
    Чтение существующего CSV файла.
    """
    if not os.path.exists(output_csv):
        return pd.DataFrame(columns=["ID", "phase", "file_name", "link"])

    try:
        return pd.read_csv(output_csv)
    except Exception as e:
        print(f"Ошибка при чтении CSV файла '{output_csv}': {e}")
        return pd.DataFrame(columns=["ID", "phase", "file_name", "link"])

def check_and_update_csv(input_excel, sheet_name, output_csv):
    """
    Проверка и обновление CSV файла с прямыми ссылками на файлы из Excel файла.

    :param input_excel: Путь к Excel файлу.
    :param sheet_name: Название листа в Excel файле.
    :param output_csv: Имя создаваемого или дополняемого файла CSV.
    """
    try:
        excel_data = extract_links_from_excel(input_excel, sheet_name)
        existing_csv_data = read_existing_csv(output_csv)

        # Проверяем последний ID в Excel и CSV
        last_excel_id = excel_data['ID пациента'].max()
        last_csv_id = existing_csv_data['ID'].max() if not existing_csv_data.empty else 0

        if last_csv_id == last_excel_id:
            print("Новых записей в Excel нет.")
            return
        elif last_csv_id > last_excel_id:
            print("Проверьте direct_links.csv, в нем лишние записи.")
            return

        # Отбираем только новые записи
        new_records = excel_data[excel_data['ID пациента'] > last_csv_id]

        print(f"Добавление новых записей в CSV файл '{output_csv}'...")

        with open(output_csv, 'a', encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")

            # Если файл новый, добавляем заголовок
            if os.stat(output_csv).st_size == 0:
                writer.writerow(["ID", "phase", "file_name", "link"])

            for _, row in tqdm(new_records.iterrows(), leave=True, total=len(new_records)):
                href = row['Местоположение файлов']
                try:
                    resource_info = get_resource_info(href)
                    if '_embedded' in resource_info:
                        items = resource_info['_embedded']['items']
                        for item in items:
                            if item['type'] == 'file':
                                folder = item['name'].split("_")[0][2:] if "_" in item['name'] else "Unknown"
                                if "_" in item['name']:
                                    parts = item['name'].split("_")
                                    if len(parts) > 1:
                                        phase = parts[1].split(".")[0]  # Извлекаем всё до точки
                                    else:
                                        phase = "Unknown"
                                else:
                                    phase = "Unknown"

                                filename = item['name'][:-4]
                                download_link = item['file'] if 'file' in item else None

                                if download_link:
                                    writer.writerow([row['ID пациента'], phase, filename, download_link])
                except Exception as e:
                    print(f"Ошибка обработки ссылки {href}: {e}")

        print(f"Новые записи загружены в '{output_csv}'.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

def create_folder_structure():
    base_path = os.path.dirname(os.path.abspath(__file__))

    file_types = ["videos", "masks"]
    model_type = ["classification", "segmentation"]
    diagnosis = ['benign', 'malignant', 'indeterminate']

    # for exam in examination:
    for type in model_type:
        if type == 'classification':
            for diag in diagnosis:
                for filee in file_types:
                    folder_path = os.path.join(base_path, 'data', type, diag, filee)
                    os.makedirs(folder_path, exist_ok=True)
        else:
            for filee in file_types:
                folder_path = os.path.join(base_path, 'data', type, filee)
                os.makedirs(folder_path, exist_ok=True)

    print(f"Структура папок успешно создана в: {os.path.join(base_path, 'data')}")

format_base = {"id": [],
               "type": [],
               "video_path": [],
               "mask_path": [],
               "diagnosis": [],
               "localization": [],
               "phase": []}

format_classification = format_base
format_segmentation = format_base

def convert_video_to_npy(video_path, output_folder):
    """
    Конвертирует видео файл в .npy массив кадров в grayscale.

    :param video_path: Путь к видео файлу
    :param output_folder: Папка для сохранения .npy файла
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео файл {video_path}")
        return

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Конвертируем в grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_resized = cv2.resize(gray_frame, (224, 224), interpolation=cv2.INTER_AREA)
        frames.append(gray_frame_resized)

    cap.release()

    frames = frames[1::3]
    if len(frames) == 0:
        print("Ошибка: не удалось прочитать ни одного кадра из видео")
        return

    frames_array = np.array(frames)

    # Создаем имя для .npy файла на основе имени видео
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    npy_filename = f"{base_name}.npy"
    npy_path = os.path.join(output_folder, npy_filename)

    # Сохраняем .npy файл
    np.save(npy_path, frames_array)
    # print(f"Видео успешно конвертировано в {npy_filename}")
    # print(f"Размер массива: {frames_array.shape} (кадры, высота, ширина)")

def download_and_convert_to_npy(file_name, download_link, download_folder):
    """
    Скачивает файл по указанной ссылке, конвертирует в .npy и удаляет оригинал.
    Защита от неверного имени файла (nan, float и т.п.) и от отсутствия ссылки.
    """
    # Защита от неверного имени файла
    if not isinstance(file_name, str) or not file_name or pd.isna(file_name):
        print(f"Пропускаю скачивание — неверное имя файла: {file_name}")
        return

    # Защита от отсутствующей/некорректной ссылки
    if not isinstance(download_link, str) or not download_link or pd.isna(download_link):
        print(f"Пропускаю скачивание {file_name} — отсутствует или неверная ссылка: {download_link}")
        return

    # Добавляем расширение .mp4 если отсутствует
    if not file_name.lower().endswith('.mp4'):
        file_name += '.mp4'

    if not os.path.exists(download_folder):
        print("Такой папки нет!")
        return

    # Скачиваем файл — с обработкой исключений сети
    file_path = os.path.join(download_folder, file_name)
    try:
        response = requests.get(download_link, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при подключении к {download_link} для файла {file_name}: {e}")
        return

    if response.status_code == 200:
        with open(file_path, 'wb') as output_file:
            output_file.write(response.content)
        # Конвертируем видео в .npy
        convert_video_to_npy(file_path, download_folder)

        # Удаляем оригинальный .mp4 файл
        try:
            os.remove(file_path)
        except OSError:
            pass
    else:
        print(f"Ошибка при скачивании файла {file_name}: {response.status_code} — пропускаю.")

def type_loader(type, row, base_path, format_file):
    if pd.notna(row['Ссылка на Файл c нативной фазой']) and pd.notna(row['Ссылка на Файл с разметкой нативной фазы']):
        videopath_cl_in_dataset = base_path + type + "/videos"
        download_and_convert_to_npy(
            file_name=row['Файл c нативной фазой'],
            download_link=row['Ссылка на Файл c нативной фазой'],
            download_folder=videopath_cl_in_dataset)

        maskpath_cl_in_dataset = base_path + type + "/masks"
        download_and_convert_to_npy(
            file_name=row['Файл с разметкой нативной фазы'],
            download_link=row['Ссылка на Файл с разметкой нативной фазы'],
            download_folder=maskpath_cl_in_dataset)

        format_file['id'].append(row['ID пациента'])
        format_file['type'].append(type)
        format_file['video_path'].append(videopath_cl_in_dataset + "/" + row['Файл c нативной фазой'] + ".npy")
        format_file['mask_path'].append(maskpath_cl_in_dataset + "/" + row['Файл с разметкой нативной фазы'] + ".npy")
        format_file['diagnosis'].append(row['Вероятный диагноз по КТ'])
        format_file['localization'].append(row['Локализация надпочечника (слева/справа)'])
        format_file['phase'].append("native")

    if pd.notna(row['Ссылка на Файл c артериальной фазой']) and pd.notna(row['Ссылка на Файл c разметкой артериальной фазы']):
        videopath_cl_in_dataset = base_path + type + "/videos"
        download_and_convert_to_npy(
            file_name=row['Файл c артериальной фазой'],
            download_link=row['Ссылка на Файл c артериальной фазой'],
            download_folder=videopath_cl_in_dataset)

        maskpath_cl_in_dataset = base_path + type + "/masks"
        download_and_convert_to_npy(
            file_name=row['Файл c разметкой артериальной фазы'],
            download_link=row['Ссылка на Файл c разметкой артериальной фазы'],
            download_folder=maskpath_cl_in_dataset)

        format_file['id'].append(row['ID пациента'])
        format_file['type'].append(type)
        format_file['video_path'].append(videopath_cl_in_dataset + "/" + row['Файл c артериальной фазой'] + ".npy")
        format_file['mask_path'].append(maskpath_cl_in_dataset + "/" + row['Файл c разметкой артериальной фазы'] + ".npy")
        format_file['diagnosis'].append(row['Вероятный диагноз по КТ'])
        format_file['localization'].append(row['Локализация надпочечника (слева/справа)'])
        format_file['phase'].append("arterial")

    if pd.notna(row['Ссылка на Файл c венозной фазой']) and pd.notna(row['Ссылка на Файл c разметкой венозной фазы']):
        videopath_cl_in_dataset = base_path + type + "/videos"
        download_and_convert_to_npy(
            file_name=row['Файл c венозной фазой'],
            download_link=row['Ссылка на Файл c венозной фазой'],
            download_folder=videopath_cl_in_dataset)

        maskpath_cl_in_dataset = base_path + type + "/masks"
        download_and_convert_to_npy(
            file_name=row['Файл c разметкой венозной фазы'],
            download_link=row['Ссылка на Файл c разметкой венозной фазы'],
            download_folder=maskpath_cl_in_dataset)

        format_file['id'].append(row['ID пациента'])
        format_file['type'].append(type)
        format_file['video_path'].append(videopath_cl_in_dataset + "/" + row['Файл c венозной фазой'] + ".npy")
        format_file['mask_path'].append(maskpath_cl_in_dataset + "/" + row['Файл c разметкой венозной фазы'] + ".npy")
        format_file['diagnosis'].append(row['Вероятный диагноз по КТ'])
        format_file['localization'].append(row['Локализация надпочечника (слева/справа)'])
        format_file['phase'].append("venous")

    if pd.notna(row['Ссылка на Файл c отсроченной фазой']) and pd.notna(row['Ссылка на Файл c разметкой отсроченной фазы']):
        videopath_cl_in_dataset = base_path + type + "/videos"
        download_and_convert_to_npy(
            file_name=row['Файл c отсроченной фазой'],
            download_link=row['Ссылка на Файл c отсроченной фазой'],
            download_folder=videopath_cl_in_dataset)

        maskpath_cl_in_dataset = base_path + type + "/masks"
        download_and_convert_to_npy(
            file_name=row['Файл c разметкой отсроченной фазы'],
            download_link=row['Ссылка на Файл c разметкой отсроченной фазы'],
            download_folder=maskpath_cl_in_dataset)

        format_file['id'].append(row['ID пациента'])
        format_file['type'].append(type)
        format_file['video_path'].append(videopath_cl_in_dataset + "/" + row['Файл c отсроченной фазой'] + ".npy")
        format_file['mask_path'].append(maskpath_cl_in_dataset + "/" + row['Файл c разметкой отсроченной фазы'] + ".npy")
        format_file['diagnosis'].append(row['Вероятный диагноз по КТ'])
        format_file['localization'].append(row['Локализация надпочечника (слева/справа)'])
        format_file['phase'].append("delay")

def excel_to_dataset(excel_file, links_csv_file):
    """
    Перенос данных из Excel-файла в коллекцию MongoDB с добавлением поля с локальным путем в файловой системе, а также ссылкой на скачивание.

    :param excel_file: Путь к Excel-файлу.
    :param links_csv_file: Путь к CSV файлу с прямыми ссылками.
    """
    df = pd.read_excel(excel_file)
    direct_links_csv = pd.read_csv(links_csv_file)

    # Добавление колонки с адресом расположения в файловой системе локальной машины
    def generate_local_path(row):
        side = "left_adrenal" if row['Локализация надпочечника (слева/справа)'] == "слева" else "right_adrenal"
        return f"data/{side}/class_{row['Доброкачественный КТ фенотип']}_{row['Неопределенный КТ фенотип']}_{row['Злокачественный КТ фенотип']}"

    df['Локальный путь'] = df.apply(generate_local_path, axis=1)

    # Добавление ссылок из CSV файла в новые поля
    phase_columns = [
        "Файл c нативной фазой", "Файл с разметкой нативной фазы",
        "Файл c артериальной фазой", "Файл c разметкой артериальной фазы",
        "Файл c венозной фазой", "Файл c разметкой венозной фазы",
        "Файл c отсроченной фазой", "Файл c разметкой отсроченной фазы"
    ]
    for phase_column in phase_columns:
        link_column_name = f"Ссылка на {phase_column}"
        df[link_column_name] = pd.Series(dtype="object") #np.nan


    for idx, row in df.iterrows():
        patient_id = row["ID пациента"]

        for phase_column in phase_columns:
            if pd.notna(row[phase_column]):  # Проверка на непустое значение
                file_name = row[phase_column]

                match = direct_links_csv[    # Ищем совпадение в csv-файле
                    (direct_links_csv["ID"] == patient_id) &
                    (direct_links_csv["file_name"] == file_name)
                    ]
                if not match.empty:
                    link_column_name = f"Ссылка на {phase_column}"
                    df.at[idx, link_column_name] = match.iloc[0]["link"]

    # print(df.columns)
    df.to_csv('dataframe.csv', index=False)

    base_path_classification = "data/classification"
    base_path_segmentation = "data/segmentation"

    for idx, row in tqdm(df.iterrows(), leave=True, total=len(df)):
        try:
            # Классификация
            if row['Доброкачественный КТ фенотип'] == 1:
                type = "/benign"
                type_loader(type, row, base_path_classification, format_classification)

            if row['Неопределенный КТ фенотип'] == 1:
                type = "/indeterminate"
                type_loader(type, row, base_path_classification, format_classification)


            if row['Злокачественный КТ фенотип'] == 1:
                type = "/malignant"
                type_loader(type, row, base_path_classification, format_classification)

            # # Сегментация
            type_loader('', row, base_path_segmentation, format_segmentation)

        except Exception as e:
            print(f"ID {idx} Не удалось скачать {row['ID пациента']}: {e}")


    with open('data/classification/format.json', 'w', encoding='utf-8') as f:
        json.dump(format_classification, f, ensure_ascii=False, indent=4)

    with open('data/segmentation/format.json', 'w', encoding='utf-8') as f:
        json.dump(format_segmentation, f, ensure_ascii=False, indent=4)

def clean_format(format_file):
    df = pd.read_json(format_file)

    format_type = format_file.split('/')[1]

    path = "data/" + format_type

    idx_list = []
    for idx, row in df.iterrows():

        detection_list = row['video_path'].split('/')
        if format_type in detection_list:
            idx_list.append(idx)

    if os.path.exists(path + "format.json"):
        os.remove(path + "format.json")

    df_write = df.iloc[idx_list]
    df_write.to_json(format_file, orient='records', force_ascii=False, indent=4)

if os.path.exists("direct_links.csv"):
    os.remove("direct_links.csv")

if os.path.exists("dataframe.csv"):
    os.remove("dataframe.csv")

print("Предварительная подготовка завершена")

input_excel = 'База данных МСКТ надпочечников_MP4.xlsx'  # Укажите путь к вашему Excel файлу
sheet_name = 'Лист1'  # Укажите имя листа в Excel
output_csv = 'direct_links.csv'  # Имя выходного файла CSV

check_and_update_csv(input_excel, sheet_name, output_csv)

create_folder_structure()

excel_base_path = os.path.join(os.path.dirname(__file__), r'База данных МСКТ надпочечников_MP4.xlsx')
links_csv_path = os.path.join(os.path.dirname(__file__), r'direct_links.csv')

excel_to_dataset(excel_file=excel_base_path, links_csv_file=links_csv_path)

clean_format("data/classification/format.json")
clean_format("data/segmentation/format.json")

shutil.make_archive('data', 'zip', 'data')

