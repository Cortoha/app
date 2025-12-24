import sys
import os
import json
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui

from app import WhaleVideoDetector, WhalePhotoIdentifier
import torch


def get_resource_path(relative_path):
    """Получить абсолютный путь к ресурсу, работает для dev и PyInstaller"""
    try:
        # PyInstaller создает временную папку и сохраняет путь в _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


class DetectionWorker(QtCore.QObject):
    """Worker object to run detection in a separate thread."""

    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    log = QtCore.pyqtSignal(str)

    def __init__(self, mode, file_path, output_dir, device_str, start_time=None, end_time=None):
        super().__init__()
        self.mode = mode  # 'photo' or 'video'
        self.file_path = file_path
        self.output_dir = output_dir
        self.device_str = device_str
        self.start_time = start_time
        self.end_time = end_time

    @QtCore.pyqtSlot()
    def run(self):
        """Run detection logic."""
        try:
            device = torch.device(self.device_str)
            # Используем функцию для определения путей к ресурсам
            yolo_path = get_resource_path('yolo/best.pt')
            embeddings_path = get_resource_path('embeddings/all_embeddings.npy')
            labels_path = get_resource_path('embeddings/all_labels.npy')
            checkpoint_path = get_resource_path('models/best_model_whale.pth')

            if self.mode == 'photo':
                self.log.emit("Загрузка моделей для фото...")
                identifier = WhalePhotoIdentifier(
                    yolo_path=yolo_path,
                    embeddings_path=embeddings_path,
                    labels_path=labels_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
                self.log.emit("Запуск идентификации кита на фото...")
                results = identifier.identify_photo(self.file_path, self.output_dir)
                self.finished.emit(
                    {
                        "mode": "photo",
                        "results": results,
                        "output_dir": str(self.output_dir),
                    }
                )
            else:
                self.log.emit("Загрузка моделей для видео...")
                detector = WhaleVideoDetector(
                    yolo_path=yolo_path,
                    embeddings_path=embeddings_path,
                    labels_path=labels_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
                self.log.emit("Запуск обработки видео...")
                start = float(self.start_time) if self.start_time else None
                end = float(self.end_time) if self.end_time else None
                detections = detector.process_video(
                    video_path=self.file_path,
                    output_dir=self.output_dir,
                    start_time=start,
                    end_time=end,
                )
                self.finished.emit(
                    {
                        "mode": "video",
                        "results": detections,
                        "output_dir": str(self.output_dir),
                    }
                )
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whale ID – Детекция и идентификация китов")
        self.resize(1000, 700)

        self.worker_thread = None
        self.worker = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        # Mode selection
        mode_group = QtWidgets.QGroupBox("Режим")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self.radio_photo = QtWidgets.QRadioButton("Фото")
        self.radio_video = QtWidgets.QRadioButton("Видео")
        self.radio_photo.setChecked(True)
        mode_layout.addWidget(self.radio_photo)
        mode_layout.addWidget(self.radio_video)
        mode_layout.addStretch()

        # File selection
        file_group = QtWidgets.QGroupBox("Входной файл")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_button = QtWidgets.QPushButton("Выбрать...")
        self.file_button.clicked.connect(self.choose_file)
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.file_button)

        # Output directory
        out_group = QtWidgets.QGroupBox("Папка для результатов")
        out_layout = QtWidgets.QHBoxLayout(out_group)
        self.out_edit = QtWidgets.QLineEdit(str(Path.cwd() / "results"))
        self.out_button = QtWidgets.QPushButton("Выбрать папку...")
        self.out_button.clicked.connect(self.choose_output_dir)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(self.out_button)

        # Video time range (only visible for video mode)
        self.video_time_group = QtWidgets.QGroupBox("Интервал времени (секунды)")
        video_time_layout = QtWidgets.QHBoxLayout(self.video_time_group)
        video_time_layout.addWidget(QtWidgets.QLabel("Старт:"))
        self.start_time_edit = QtWidgets.QLineEdit()
        self.start_time_edit.setPlaceholderText("0")
        video_time_layout.addWidget(self.start_time_edit)
        video_time_layout.addWidget(QtWidgets.QLabel("Конец:"))
        self.end_time_edit = QtWidgets.QLineEdit()
        self.end_time_edit.setPlaceholderText("весь видео")
        video_time_layout.addWidget(self.end_time_edit)
        video_time_layout.addStretch()
        self.video_time_group.setVisible(False)
        self.radio_video.toggled.connect(lambda checked: self.video_time_group.setVisible(checked))

        # Device selection
        device_group = QtWidgets.QGroupBox("Устройство")
        device_layout = QtWidgets.QHBoxLayout(device_group)
        self.device_combo = QtWidgets.QComboBox()
        if torch.cuda.is_available():
            self.device_combo.addItems(["cuda", "cpu"])
        else:
            self.device_combo.addItems(["cpu"])
        device_layout.addWidget(QtWidgets.QLabel("Вычисления:"))
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()

        # Run button
        self.run_button = QtWidgets.QPushButton("Запустить обработку")
        self.run_button.setFixedHeight(40)
        self.run_button.clicked.connect(self.start_detection)

        # Splitter for results and logs
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left: image / results
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        self.image_label = QtWidgets.QLabel("Здесь будет показан результат (изображение).")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setFrameShape(QtWidgets.QFrame.StyledPanel)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            ["Ранг", "ID кита", "Count"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.table)

        # Right: logs
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.addWidget(QtWidgets.QLabel("Логи:"))
        self.log_edit = QtWidgets.QTextEdit()
        self.log_edit.setReadOnly(True)
        right_layout.addWidget(self.log_edit)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(mode_group)
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.video_time_group)
        main_layout.addWidget(out_group)
        main_layout.addWidget(device_group)
        main_layout.addWidget(self.run_button)
        main_layout.addWidget(splitter)

    def append_log(self, text: str):
        self.log_edit.append(text)
        self.log_edit.ensureCursorVisible()

    def choose_file(self):
        if self.radio_photo.isChecked():
            filter_str = "Изображения (*.jpg *.jpeg *.png *.bmp *.gif *.webp)"
        else:
            filter_str = "Видео (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)"

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выбор файла", "", filter_str
        )
        if path:
            self.file_edit.setText(path)

    def choose_output_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выбор папки для результатов", ""
        )
        if path:
            self.out_edit.setText(path)

    def start_detection(self):
        file_path = self.file_edit.text().strip()
        out_dir = self.out_edit.text().strip()

        if not file_path:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Выберите входной файл.")
            return
        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Файл не найден.")
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        mode = "photo" if self.radio_photo.isChecked() else "video"
        device_str = self.device_combo.currentText()

        start_time = self.start_time_edit.text().strip() if mode == "video" else None
        end_time = self.end_time_edit.text().strip() if mode == "video" else None
        start_time = start_time if start_time else None
        end_time = end_time if end_time else None

        self.run_button.setEnabled(False)
        self.append_log(f"Режим: {mode}, файл: {file_path}")
        if mode == "video" and (start_time or end_time):
            self.append_log(f"Интервал: {start_time or 'начало'} - {end_time or 'конец'}")
        self.append_log("Запуск обработки...")

        # Clear previous results
        self.table.setRowCount(0)
        self.image_label.setPixmap(QtGui.QPixmap())
        self.image_label.setText("Обработка запущена...")

        self.worker_thread = QtCore.QThread()
        self.worker = DetectionWorker(mode, file_path, str(out_path), device_str, start_time, end_time)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker.failed.connect(self.worker_thread.quit)

        self.worker_thread.start()

    def on_detection_finished(self, data: dict):
        try:
            mode = data.get("mode")
            output_dir = Path(data.get("output_dir", "."))
            self.append_log("Обработка завершена.")

            if mode == "photo":
                img_path = None
                # Ожидаем имя файла, как в WhalePhotoIdentifier
                for f in output_dir.glob("whale_*_identification.jpg"):
                    img_path = f
                    break
                if img_path and img_path.exists():
                    pixmap = QtGui.QPixmap(str(img_path))
                    self.image_label.setPixmap(
                        pixmap.scaled(
                            self.image_label.size(),
                            QtCore.Qt.KeepAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )
                else:
                    self.image_label.setText(
                        "Результирующее изображение не найдено."
                    )

                # Загрузим JSON результатов, если он есть
                json_path = output_dir / "photo_identification_results.json"
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        self.fill_photo_table(results)
                    except Exception as e:
                        self.append_log(f"Ошибка чтения JSON результатов: {e}")

            else:
                # Video mode
                img_path = output_dir / "whale_detection_result.jpg"
                if img_path.exists():
                    pixmap = QtGui.QPixmap(str(img_path))
                    self.image_label.setPixmap(
                        pixmap.scaled(
                            self.image_label.size(),
                            QtCore.Qt.KeepAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )
                else:
                    self.image_label.setText(
                        "Результирующее изображение не найдено."
                    )

                # Загрузим JSON результатов по видео
                json_path = output_dir / "detection_results.json"
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        self.fill_video_table(results)
                    except Exception as e:
                        self.append_log(f"Ошибка чтения JSON результатов: {e}")

        finally:
            self.run_button.setEnabled(True)

    def on_detection_failed(self, error: str):
        QtWidgets.QMessageBox.critical(self, "Ошибка обработки", error)
        self.append_log(f"Ошибка: {error}")
        self.run_button.setEnabled(True)

    def fill_photo_table(self, results):
        """Заполнить таблицу для фото по данным JSON."""
        # Показываем только TOP-5 для первого кита (обычно он один на фото)
        if not results:
            return
        first = results[0]
        top5 = first.get("top_5_whales", [])[:5]

        self.table.setRowCount(len(top5))
        for row, item in enumerate(top5):
            self.table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(str(item.get("rank")))
            )
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(item.get("whale_id")))
            )
            self.table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(str(item.get("count")))
            )

    def fill_video_table(self, results):
        """Заполнить таблицу для видео по JSON."""
        # Преобразуем словарь {whale_id: {...}} в список, сортированный по best_count
        items = []
        for whale_id, info in results.items():
            items.append(
                (whale_id, int(info.get("best_count", 0)), int(info.get("detections", 0)))
            )
        items.sort(key=lambda x: x[1], reverse=True)

        self.table.setRowCount(len(items))
        for idx, (whale_id, best_count, detections) in enumerate(items, start=1):
            self.table.setItem(idx - 1, 0, QtWidgets.QTableWidgetItem(str(idx)))
            self.table.setItem(
                idx - 1, 1, QtWidgets.QTableWidgetItem(str(whale_id))
            )
            self.table.setItem(
                idx - 1, 2, QtWidgets.QTableWidgetItem(str(best_count))
            )


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



