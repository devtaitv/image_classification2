import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import time
import os
from skimage.feature import hog
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import joblib
from sklearn.datasets import load_iris
import math


class ID3Classifier:
    def __init__(self):
        self.tree = None
        self.features = None

    def entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            entropy -= p * np.log2(p) if p > 0 else 0
        return entropy

    def information_gain(self, X, y, feature):
        entropy_parent = self.entropy(y)
        values = np.unique(X[:, feature])
        weighted_entropy = 0
        for value in values:
            subset_indices = X[:, feature] == value
            subset_y = y[subset_indices]
            weight = len(subset_y) / len(y)
            weighted_entropy += weight * self.entropy(subset_y)
        return entropy_parent - weighted_entropy

    def build_tree(self, X, y, features, depth=0, max_depth=10):
        if len(np.unique(y)) == 1:
            return {'type': 'leaf', 'value': y[0]}
        if depth >= max_depth or len(features) == 0:
            return {'type': 'leaf', 'value': np.bincount(y).argmax()}

        gains = [self.information_gain(X, y, f) for f in range(len(features))]
        best_feature_idx = np.argmax(gains)
        best_feature = features[best_feature_idx]

        tree = {'type': 'node', 'feature': best_feature, 'children': {}}
        remaining_features = features[:best_feature_idx] + features[best_feature_idx + 1:]

        for value in np.unique(X[:, best_feature_idx]):
            subset_indices = X[:, best_feature_idx] == value
            subset_X = X[subset_indices]
            subset_y = y[subset_indices]
            if len(subset_X) == 0:
                tree['children'][value] = {'type': 'leaf', 'value': np.bincount(y).argmax()}
            else:
                tree['children'][value] = self.build_tree(subset_X, subset_y, remaining_features, depth + 1, max_depth)

        return tree

    def fit(self, X, y):
        self.features = list(range(X.shape[1]))
        self.tree = self.build_tree(X, y, self.features)
        return self

    def predict_one(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['value']
        feature = tree['feature']
        value = x[feature]
        if value not in tree['children']:
            return max(set(tree['children'].values()), key=list(tree['children'].values()).count)
        return self.predict_one(x, tree['children'][value])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Phân loại Hình ảnh")
        self.root.geometry("1200x800")

        # Khởi tạo các biến
        self.models = {
            'Naive Bayes': GaussianNB(),
            'CART': DecisionTreeClassifier(criterion='gini'),
            'ID3': ID3Classifier(),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        }
        self.current_model = None
        self.X, self.y = None, None
        self.class_labels = None
        self.dataset_path = None

        # Thiết lập style
        style = ttk.Style()
        style.configure('Custom.TButton', padding=5)

        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Cấu hình grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=3)
        main_frame.rowconfigure(0, weight=1)

        self.setup_left_frame(main_frame)
        self.setup_middle_frame(main_frame)
        self.setup_right_frame(main_frame)

    def setup_left_frame(self, main_frame):
        left_frame = ttk.LabelFrame(main_frame, text="Training Model", padding="10")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.N, tk.S))

        # Dataset controls
        dataset_frame = ttk.LabelFrame(left_frame, text="Dataset", padding="5")
        dataset_frame.grid(row=0, column=0, pady=5, sticky='ew')

        ttk.Button(dataset_frame, text="Chọn thư mục dataset",
                   command=self.load_dataset, style='Custom.TButton').pack(fill='x', pady=2)
        ttk.Button(dataset_frame, text="Load IRIS Dataset",
                   command=self.load_iris_dataset, style='Custom.TButton').pack(fill='x', pady=2)

        self.dataset_label = ttk.Label(dataset_frame, text="Dataset: Chưa chọn")
        self.dataset_label.pack(fill='x', pady=2)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(dataset_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x', pady=2)

        self.progress_label = ttk.Label(dataset_frame, text="0%")
        self.progress_label.pack(pady=2)

        # Model controls
        model_frame = ttk.LabelFrame(left_frame, text="Model Configuration", padding="5")
        model_frame.grid(row=1, column=0, pady=5, sticky='ew')

        ttk.Label(model_frame, text="Thuật toán:").pack(fill='x', pady=2)
        self.model_var = tk.StringVar(value='Naive Bayes')
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=list(self.models.keys()))
        model_combo.pack(fill='x', pady=2)

        ttk.Label(model_frame, text="Tỷ lệ train-test:").pack(fill='x', pady=2)
        self.split_var = tk.StringVar(value='80-20')
        split_combo = ttk.Combobox(model_frame, textvariable=self.split_var,
                                   values=['80-20', '70-30', '60-40'])
        split_combo.pack(fill='x', pady=2)

        # Training buttons
        ttk.Button(model_frame, text="Train Model",
                   command=self.train_model, style='Custom.TButton').pack(fill='x', pady=2)
        ttk.Button(model_frame, text="Lưu Model",
                   command=self.save_model, style='Custom.TButton').pack(fill='x', pady=2)
        ttk.Button(model_frame, text="Tải Model",
                   command=self.load_model, style='Custom.TButton').pack(fill='x', pady=2)

        # Image prediction frame
        predict_frame = ttk.LabelFrame(left_frame, text="Dự đoán ảnh mới", padding="5")
        predict_frame.grid(row=2, column=0, pady=5, sticky='ew')

        ttk.Button(predict_frame, text="Chọn ảnh để dự đoán",
                   command=self.predict_new_image, style='Custom.TButton').pack(fill='x', pady=2)
        ttk.Button(predict_frame, text="Thêm ảnh mới vào dataset",
                   command=self.add_new_image, style='Custom.TButton').pack(fill='x', pady=2)

        # Preview frame
        self.preview_frame = ttk.LabelFrame(left_frame, text="Xem trước ảnh", padding="5")
        self.preview_frame.grid(row=3, column=0, pady=5, sticky='ew')

        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(pady=2)

    def load_iris_dataset(self):
        try:
            iris = load_iris()
            self.X = iris.data
            self.y = iris.target
            self.class_labels = [str(i) for i in range(3)]

            self.dataset_label.config(text="Dataset: IRIS")
            self.metrics_text.insert(tk.END, "\nĐã load IRIS dataset:\n")
            self.metrics_text.insert(tk.END, f"- Số mẫu: {len(self.X)}\n")
            self.metrics_text.insert(tk.END, f"- Số đặc trưng: {self.X.shape[1]}\n")
            self.metrics_text.insert(tk.END, f"- Số lớp: {len(np.unique(self.y))}\n")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi load IRIS dataset: {str(e)}")

    # [Rest of the methods remain the same as in the original code, just remove references to SVM and KNN]

    def setup_middle_frame(self, main_frame):
        middle_frame = ttk.LabelFrame(main_frame, text="Kết quả & Metrics", padding="10")
        middle_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.N, tk.S))

        # Text widget với scrollbar
        self.metrics_text = tk.Text(middle_frame, wrap='word', width=40, height=30)
        scrollbar = ttk.Scrollbar(middle_frame, orient='vertical', command=self.metrics_text.yview)
        self.metrics_text['yscrollcommand'] = scrollbar.set

        self.metrics_text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(0, weight=1)

    def setup_right_frame(self, main_frame):
        right_frame = ttk.LabelFrame(main_frame, text="Biểu đồ & Visualization", padding="10")
        right_frame.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Notebook để chứa nhiều biểu đồ
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True)

        # Tab cho metrics
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text='Metrics')

        self.fig_metrics = Figure(figsize=(6, 4), dpi=100)
        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, master=metrics_tab)
        self.canvas_metrics.get_tk_widget().pack(fill='both', expand=True)

        # Tab cho confusion matrix
        cm_tab = ttk.Frame(self.notebook)
        self.notebook.add(cm_tab, text='Confusion Matrix')

        self.fig_cm = Figure(figsize=(6, 4), dpi=100)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=cm_tab)
        self.canvas_cm.get_tk_widget().pack(fill='both', expand=True)

    def update_progress(self, current, total):
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{int(progress)}%")
        self.root.update_idletasks()

    def load_dataset(self):
        try:
            self.dataset_path = filedialog.askdirectory(title="Chọn thư mục dataset")
            if not self.dataset_path:
                return

            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "Đang load dataset...\n")

            X, y = [], []
            image_files = []
            class_counts = {}

            # Tìm tất cả các file ảnh hợp lệ
            for class_folder in os.listdir(self.dataset_path):
                class_path = os.path.join(self.dataset_path, class_folder)
                if os.path.isdir(class_path):
                    class_images = [f for f in os.listdir(class_path)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    class_counts[class_folder] = len(class_images)
                    for image_file in class_images:
                        image_files.append((os.path.join(class_path, image_file), class_folder))

            if not image_files:
                raise ValueError("Không tìm thấy ảnh trong thư mục!")

            # Cập nhật progress bar
            total_images = len(image_files)
            self.progress_bar['maximum'] = total_images

            # Load và xử lý từng ảnh
            for i, (image_path, class_label) in enumerate(image_files):
                try:
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize((64, 64))
                    features = self.extract_features(np.array(img))
                    X.append(features)
                    y.append(class_label)
                    self.update_progress(i + 1, total_images)
                except Exception as e:
                    self.metrics_text.insert(tk.END, f"Lỗi xử lý ảnh {image_path}: {str(e)}\n")

            self.X, self.y = np.array(X), np.array(y)
            self.class_labels = sorted(set(self.y))

            # Hiển thị thông tin dataset
            self.dataset_label.config(text=f"Dataset: {os.path.basename(self.dataset_path)}")
            self.metrics_text.insert(tk.END, f"\nĐã load xong dataset:\n")
            self.metrics_text.insert(tk.END, f"- Tổng số ảnh: {len(self.X)}\n")
            self.metrics_text.insert(tk.END, "- Phân bố các lớp:\n")
            for class_name, count in class_counts.items():
                self.metrics_text.insert(tk.END, f"  + {class_name}: {count} ảnh\n")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi load dataset: {str(e)}")
            self.metrics_text.insert(tk.END, f"Lỗi: {str(e)}\n")

    def extract_features(self, image):
        if image.ndim == 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)
        return features

    def train_model(self):
        try:
            if self.X is None or len(self.X) == 0:
                raise ValueError("Vui lòng load dataset trước!")

            model_name = self.model_var.get()
            self.current_model = self.models[model_name]

            self.metrics_text.insert(tk.END, f"\nĐang train model {model_name}...\n")

            # Chia dữ liệu theo tỷ lệ đã chọn
            train_size = float(self.split_var.get().split('-')[0]) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, train_size=train_size, random_state=42, stratify=self.y)

            # Training
            start_time = time.time()
            self.current_model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predictions và metrics
            y_pred = self.current_model.predict(X_test)
            y_pred_proba = self.current_model.predict_proba(X_test)

            # Tính toán các metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred, labels=self.class_labels)

            # Hiển thị kết quả
            self.metrics_text.insert(tk.END, f"\nKết quả training:\n")
            self.metrics_text.insert(tk.END, f"- Thời gian training: {train_time:.2f}s\n")
            self.metrics_text.insert(tk.END, f"- Accuracy: {accuracy:.4f}\n")
            self.metrics_text.insert(tk.END, f"- Precision: {precision:.4f}\n")
            self.metrics_text.insert(tk.END, f"- Recall: {recall:.4f}\n")

            # Cập nhật biểu đồ
            self.update_metrics_chart(accuracy, precision, recall)
            self.update_confusion_matrix(cm)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi");

    def update_metrics_chart(self, accuracy, precision, recall):
        self.fig_metrics.clear()
        ax = self.fig_metrics.add_subplot(111)

        metrics = ['Accuracy', 'Precision', 'Recall']
        values = [accuracy, precision, recall]
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Giá trị')
        ax.set_title('Đánh giá Model')

        # Thêm giá trị lên các cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        self.canvas_metrics.draw()

    def update_confusion_matrix(self, cm):
        self.fig_cm.clear()
        ax = self.fig_cm.add_subplot(111)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_labels,
                    yticklabels=self.class_labels, ax=ax)

        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Thực tế')
        ax.set_xlabel('Dự đoán')

        self.fig_cm.tight_layout()
        self.canvas_cm.draw()

    def save_model(self):
        if self.current_model is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng train model trước khi lưu!")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
            )

            if file_path:
                # Lưu model và thông tin bổ sung
                model_info = {
                    'model': self.current_model,
                    'class_labels': self.class_labels,
                    'model_name': self.model_var.get(),
                    'feature_params': {
                        'image_size': (64, 64),
                        'hog_params': {
                            'orientations': 8,
                            'pixels_per_cell': (16, 16),
                            'cells_per_block': (1, 1)
                        }
                    }
                }

                joblib.dump(model_info, file_path)
                self.metrics_text.insert(tk.END, f"\nĐã lưu model tại: {file_path}\n")
                messagebox.showinfo("Thành công", "Đã lưu model thành công!")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi lưu model: {str(e)}")
            self.metrics_text.insert(tk.END, f"Lỗi khi lưu model: {str(e)}\n")

    def load_model(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
            )

            if file_path:
                # Load model và thông tin
                model_info = joblib.load(file_path)

                self.current_model = model_info['model']
                self.class_labels = model_info['class_labels']
                self.model_var.set(model_info['model_name'])

                self.metrics_text.insert(tk.END, f"\nĐã tải model từ: {file_path}\n")
                self.metrics_text.insert(tk.END, f"Model type: {model_info['model_name']}\n")
                self.metrics_text.insert(tk.END, f"Số lượng lớp: {len(self.class_labels)}\n")
                messagebox.showinfo("Thành công", "Đã tải model thành công!")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tải model: {str(e)}")
            self.metrics_text.insert(tk.END, f"Lỗi khi tải model: {str(e)}\n")

    def predict_new_image(self):
        if self.current_model is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng train hoặc tải model trước!")
            return

        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
            )

            if file_path:
                # Load và xử lý ảnh
                img = Image.open(file_path).convert('RGB')
                img_display = img.copy()
                img = img.resize((64, 64))

                # Trích xuất đặc trưng
                features = self.extract_features(np.array(img))

                # Dự đoán
                prediction = self.current_model.predict([features])[0]
                probabilities = self.current_model.predict_proba([features])[0]

                # Hiển thị kết quả
                self.metrics_text.insert(tk.END, f"\nKết quả dự đoán cho ảnh: {os.path.basename(file_path)}\n")
                self.metrics_text.insert(tk.END, f"Lớp dự đoán: {prediction}\n")
                self.metrics_text.insert(tk.END, "Xác suất cho từng lớp:\n")

                for class_label, prob in zip(self.class_labels, probabilities):
                    self.metrics_text.insert(tk.END, f"- {class_label}: {prob:.4f}\n")

                # Hiển thị ảnh preview
                self.show_preview(img_display)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán ảnh: {str(e)}")
            self.metrics_text.insert(tk.END, f"Lỗi khi dự đoán ảnh: {str(e)}\n")

    def add_new_image(self):
        if not self.dataset_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục dataset trước!")
            return

        try:
            # Chọn ảnh
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
            )

            if file_path:
                # Chọn lớp
                class_dialog = tk.Toplevel(self.root)
                class_dialog.title("Chọn lớp")
                class_dialog.geometry("300x400")

                # Frame cho danh sách lớp hiện có
                list_frame = ttk.LabelFrame(class_dialog, text="Lớp hiện có", padding="5")
                list_frame.pack(fill='both', expand=True, padx=5, pady=5)

                # Listbox cho các lớp
                class_listbox = tk.Listbox(list_frame)
                class_listbox.pack(fill='both', expand=True)

                # Thêm các lớp hiện có vào listbox
                existing_classes = [d for d in os.listdir(self.dataset_path)
                                    if os.path.isdir(os.path.join(self.dataset_path, d))]
                for class_name in existing_classes:
                    class_listbox.insert(tk.END, class_name)

                # Frame cho lớp mới
                new_class_frame = ttk.LabelFrame(class_dialog, text="Thêm lớp mới", padding="5")
                new_class_frame.pack(fill='x', padx=5, pady=5)

                new_class_var = tk.StringVar()
                ttk.Entry(new_class_frame, textvariable=new_class_var).pack(fill='x', pady=2)

                def confirm_class():
                    class_name = new_class_var.get().strip()
                    if not class_name:  # Nếu không có lớp mới, lấy lớp đã chọn
                        selection = class_listbox.curselection()
                        if not selection:
                            messagebox.showwarning("Cảnh báo", "Vui lòng chọn hoặc nhập tên lớp!")
                            return
                        class_name = class_listbox.get(selection[0])

                    # Tạo thư mục lớp nếu chưa tồn tại
                    class_path = os.path.join(self.dataset_path, class_name)
                    if not os.path.exists(class_path):
                        os.makedirs(class_path)

                    # Copy ảnh vào thư mục lớp
                    dest_path = os.path.join(class_path, os.path.basename(file_path))
                    img = Image.open(file_path)
                    img.save(dest_path)

                    self.metrics_text.insert(tk.END,
                                             f"\nĐã thêm ảnh {os.path.basename(file_path)} vào lớp {class_name}\n")

                    # Hiển thị preview
                    self.show_preview(img)

                    class_dialog.destroy()

                ttk.Button(class_dialog, text="Xác nhận", command=confirm_class).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi thêm ảnh mới: {str(e)}")
            self.metrics_text.insert(tk.END, f"Lỗi khi thêm ảnh mới: {str(e)}\n")

    def show_preview(self, img):
        # Resize ảnh để hiển thị preview
        display_size = (150, 150)
        img.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # Giữ references original]
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()