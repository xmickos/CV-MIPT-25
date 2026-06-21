import h5py
import matplotlib.pyplot as plt
import os
import numpy as np

S3DIS_CLASS_NAMES = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

S3DIS_NAME_TO_ID = {v: k for k, v in S3DIS_CLASS_NAMES.items()}

def load_and_preprocess_s3dis_from_annotations(root_dir: str):
    """
    Обходит все Area_*/.../Annotations/*.txt и формирует единый массив:

        Xn Yn Zn Rn Gn Bn label

    где:
      - Xn,Yn,Zn — нормализованные координаты (центрирование + масштабирование)
      - Rn,Gn,Bn — цвета в [0, 1]
      - label    — id класса S3DIS (float в dataset, int возвращается отдельно)
    """

    all_parts = []
    all_labels = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # интересны только папки Annotations
        if os.path.basename(dirpath) != "Annotations":
            continue

        for fname in filenames:
            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(dirpath, fname)

            # имя класса до первого подчёркивания: chair_1.txt -> chair
            class_name = fname.split("_")[0]

            # если класс не в стандартном списке — считаем его clutter
            label_id = S3DIS_NAME_TO_ID.get(class_name, S3DIS_NAME_TO_ID["clutter"])

            # 1) Загрузка: X Y Z R G B
            print(f"current file: {file_path}")
            try:
                data = np.loadtxt(file_path)
            except:
                print(f"exception with {file_path}")
                continue

            if data.ndim != 2 or data.shape[1] != 6:
                print("Пропускаю странный файл (не 6 столбцов):", file_path)
                continue

            coords = data[:, 0:3].astype(np.float32)  # X Y Z
            colors = data[:, 3:6].astype(np.float32)  # R G B

            # 2) Нормализация цветов R,G,B в [0,1]
            colors_norm = np.clip(colors / 255.0, 0.0, 1.0)

            # 3) Массив меток для всех точек этого объекта
            labels = np.full((coords.shape[0], 1), label_id, dtype=np.float32)
            labels_int = np.full(coords.shape[0], label_id, dtype=np.int32)

            # 4) Формируем таблицу X Y Z R G B label через hstack
            room_table = np.hstack([
                coords,
                colors_norm,
                labels
            ]).astype(np.float32)

            all_parts.append(room_table)
            all_labels.append(labels_int)

    if not all_parts:
        raise RuntimeError("Не найдено ни одного корректного файла в Annotations")

    dataset = np.vstack(all_parts)               # (N, 7)
    labels_all = np.concatenate(all_labels)      # (N,)

    # 5) Предобработка: нормализация координат X,Y,Z
    coords = dataset[:, 0:3]
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std[std == 0] = 1.0
    coords_norm = (coords - mean) / std
    dataset[:, 0:3] = coords_norm

    print("Готовый dataset:", dataset.shape)
    return dataset.astype(np.float32), labels_all.astype(np.int32)


def save_dataset_all_formats(dataset: np.ndarray,
                             labels: np.ndarray,
                             format: str,
                             base_name: str = "s3dis_dataset"):
    """
    Сохраняет dataset и labels в одном из форматов:
      - .npy   (NumPy бинарный)
      - .txt   (человекочитаемый)
      - .h5    (HDF5, два датасета: 'dataset' и 'labels')

    dataset: (N, D)
    labels : (N,)
    """
    print("Сохранено файлы:")
    # 1) .npy
    if format == 'npy' or format == '.npy':
        np.save(base_name + ".npy", dataset)
        print("  ", base_name + ".npy")

    # 2) .txt (каждая строка — одна точка)
    if format == 'txt' or format == '.txt':
        np.savetxt(base_name + ".txt", dataset, fmt="%.6f")
        print("  ", base_name + ".txt")

    # 3) .h5
    if format == 'h5' or format == '.h5':
        with h5py.File(base_name + ".h5", "w") as f:
            f.create_dataset("dataset", data=dataset)
            f.create_dataset("labels", data=labels)
        print("  ", base_name + ".h5")


def plot_label_histogram(labels: np.ndarray, class_names: dict | None = None):
    """
    Строит гистограмму распределения меток по файлу/датасету.

    labels      : (N,) целые метки
    class_names : опционально словарь {label_id: "human_name"}
    """
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts)

    if class_names is not None:
        # Подписи классов, если переданы
        xticks = [class_names.get(int(k), str(int(k))) for k in unique]
        plt.xticks(unique, xticks, rotation=45, ha="right")
    else:
        plt.xticks(unique)

    plt.xlabel("Класс")
    plt.ylabel("Количество точек")
    plt.title("Распределение меток в датасете")
    plt.tight_layout()
    plt.show()

    plt.savefig("Figure_2.png")


root_dir = "/Users/xmickos/Desktop/MIPT/МИПТ) 9 сем)/ЦИТиС/Дьяченко/Lab6/Stanford3dDataset_v1.2"

dataset, labels = load_and_preprocess_s3dis_from_annotations(root_dir)


print("Форма dataset:", dataset.shape)
print("Первые 5 строк:\n", dataset[:5])

save_dataset_all_formats(dataset, labels, "npy")

plot_label_histogram(labels, class_names=S3DIS_CLASS_NAMES)
