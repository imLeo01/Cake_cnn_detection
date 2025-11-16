import sys, os, csv, time, numpy as np, tensorflow as tf, cv2, tempfile
from datetime import datetime
from pathlib import Path

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QScrollArea
)
from PyQt6.QtGui import QPixmap, QFont, QImage
from PyQt6.QtCore import Qt, QTimer

# optional torch / ultralytics imports (attempt to import gracefully)
try:
    import torch
except Exception as e:
    torch = None

# --- DỮ LIỆU CỐ ĐỊNH ---
CLASS_NAMES = [
    'banh chuoi nuong', 'banh cua bo', 'banh da lon', 'banh mi dua luoi',
    'cha bong cay', 'cookies dua', 'croissant', 'egg tart',
    'muffin viet quat', 'patechaud'
]
PRICE_LIST = {
    'banh chuoi nuong': 19000, 'banh cua bo': 18000, 'banh da lon': 23000,
    'banh mi dua luoi': 15000, 'cha bong cay': 27000, 'cookies dua': 23000,
    'croissant': 30000, 'egg tart': 21000, 'muffin viet quat': 25000,
    'patechaud': 30000
}
DISPLAY_NAMES = {
    'banh chuoi nuong': 'Bánh Chuối Nướng 🍌', 'banh cua bo': 'Bánh Cua Bơ 🦀',
    'banh da lon': 'Bánh Da Lợn 🐷', 'banh mi dua luoi': 'Bánh Mì Dừa Lưới 🍈',
    'cha bong cay': 'Chà Bông Cây 🌶️', 'cookies dua': 'Cookies Dừa 🥥',
    'croissant': 'Croissant 🥐', 'egg tart': 'Egg Tart 🥚',
    'muffin viet quat': 'Muffin Việt Quất 🫐', 'patechaud': 'Patechaud 🥧'
}

# (ĐÃ SỬA LỖI 1: ValueError) Sửa kích thước ảnh khớp với model
IMG_WIDTH, IMG_HEIGHT = 224, 224

MODELS_DIR = Path("models\keras2")
MODEL_PATH = MODELS_DIR / "bakery_cnn.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"          

CNN_CONF_THRESHOLD = 0.55  
YOLO_MIN_AREA = 1500     
YOLO_PADDING = 0.12     

DEBOUNCE_BUTTON_MS = 2500 

class BillItemWidget(QFrame):
    def __init__(self, class_name, price, parent):
        super().__init__()
        self.class_name = class_name
        self.price = price
        self.quantity = 1
        self.parent_app = parent
        layout = QHBoxLayout(self)
        name = DISPLAY_NAMES.get(class_name, class_name)
        self.name_label = QLabel(f"{name}\n{price:,.0f}đ")
        self.name_label.setFont(QFont("Arial", 12))
        self.qty_label = QLabel("1")
        self.qty_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.qty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_minus, btn_plus = QPushButton("–"), QPushButton("+")
        for b in (btn_minus, btn_plus): b.setFixedSize(30, 30)
        btn_minus.clicked.connect(self.decrease_quantity)
        btn_plus.clicked.connect(self.increase_quantity)
        layout.addWidget(self.name_label, 1)
        layout.addWidget(btn_minus)
        layout.addWidget(self.qty_label)
        layout.addWidget(btn_plus)

    def increase_quantity(self):
        self.quantity += 1
        self.qty_label.setText(str(self.quantity))
        self.parent_app.update_total()

    def decrease_quantity(self):
        if self.quantity > 0:
            self.quantity -= 1
            self.qty_label.setText(str(self.quantity))
            self.parent_app.update_total()

# --- APP CHÍNH ---
class BakeryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍞 Tiệm Bánh 3ITECH (AI Smart POS)")
        self.setGeometry(170, 170, 1200, 700)
        self.model = None         # tensorflow CNN model (.h5)
        self.model_pt = None      # YOLO .pt model (for cropping only)
        self.bill_items, self.last_detected_time = {}, {}
        self.realtime_enabled, self.capture = False, None
        self.current_image_path, self.current_frame = None, None
        
        # (ĐÃ THÊM LỖI 3) Biến lưu trữ phát hiện realtime
        self.current_detections = [] 

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.load_models()
        self.setup_ui()
        self.apply_style()

    # --- MODELS LOAD ---
    # (Không thay đổi)
    def load_models(self):
        # load CNN .h5
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ CNN .h5 model loaded.")
            except Exception as e:
                print("❌ Lỗi khi load CNN .h5:", e)
                self.model = None
        else:
            print("❌ Không tìm thấy CNN .h5 model ở", MODEL_PATH)

    # --- UI (ĐÃ SỬA LỖI 3) ---
    def setup_ui(self):
        main = QHBoxLayout()

        # CAMERA
        col1 = QWidget()
        c1 = QVBoxLayout(col1)
        c1.setAlignment(Qt.AlignmentFlag.AlignTop)
        logo = QLabel("🍞 3ITECH SMART POS")
        logo.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.camera_label = QLabel("📷 Chưa bật camera")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(500, 500)

        self.btn_realtime = QPushButton("📹 BẬT CAMERA REALTIME")
        self.btn_realtime.clicked.connect(self.toggle_realtime)

        btn_load = QPushButton("📂 TẢI ẢNH TĨNH")
        btn_load.clicked.connect(self.select_image)
        
        # (ĐÃ SỬA LỖI 3) Gán self.btn_detect và đổi tên nút
        self.btn_detect = QPushButton("📸 THÊM BÁNH VÀO HÓA ĐƠN")
        self.btn_detect.clicked.connect(self.run_detection)
        
        btn_pay = QPushButton("💰 THANH TOÁN")
        btn_pay.clicked.connect(self.pay_bill)

        c1.addWidget(logo)
        c1.addWidget(self.camera_label)
        c1.addWidget(self.btn_realtime)
        c1.addWidget(btn_load)
        c1.addWidget(self.btn_detect) # (ĐÃ SỬA)
        c1.addWidget(btn_pay)

        # BILL
        col2 = QWidget()
        c2 = QVBoxLayout(col2)
        title = QLabel("🧾 HÓA ĐƠN TẠM TÍNH")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.scroll = QScrollArea()
        self.scroll_widget = QWidget()
        self.bill_layout = QVBoxLayout(self.scroll_widget)
        self.bill_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.scroll_widget)
        self.scroll.setWidgetResizable(True)
        c2.addWidget(title)
        c2.addWidget(self.scroll)

        # TOTAL
        col3 = QWidget()
        c3 = QVBoxLayout(col3)
        total_title = QLabel("💰 TỔNG KẾT")
        total_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.subtotal_label = QLabel("Tổng tiền hàng:\n0đ")
        self.final_total_label = QLabel("0đ")
        self.final_total_label.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        btn_clear = QPushButton("🗑 XÓA BILL")
        btn_clear.clicked.connect(self.clear_bill)
        c3.addWidget(total_title)
        c3.addWidget(self.subtotal_label)
        c3.addWidget(self.final_total_label)
        c3.addStretch()
        c3.addWidget(btn_clear)

        main.addWidget(col1, 3)
        main.addWidget(col2, 3)
        main.addWidget(col3, 2)
        central = QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)

    # (ĐÃ SỬA LỖI 3) Thêm style cho nút
    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #FFF8E1; }
            QLabel { color: #4E342E; }
            QFrame { background-color: #FFECB3; border-radius: 8px; }
            QPushButton {
                background-color: #8D6E63; color: white;
                font-size: 14px; font-weight: bold; padding: 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #A1887F; }
            
            QPushButton[text="📸 THÊM BÁNH VÀO HÓA ĐƠN"] {
                background-color: #FF6F00;
            }
            QPushButton[text="📸 THÊM BÁNH VÀO HÓA ĐƠN"]:hover {
                background-color: #FF8F00;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)

    # --- CAMERA ---
    # (Không thay đổi)
    def toggle_realtime(self):
        if not self.realtime_enabled:
            for cam_id in [0, 1, 2]:
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    self.capture = cap
                    break
            if not self.capture or not self.capture.isOpened():
                print("⚠️ Không mở được camera!")
                return
            self.realtime_enabled = True
            self.btn_realtime.setText("⏹ TẮT CAMERA REALTIME")
            self.timer.start(33)
            print("✅ Camera bật realtime.")
        else:
            self.realtime_enabled = False
            self.btn_realtime.setText("📹 BẬT CAMERA REALTIME")
            self.timer.stop()
            if self.capture:
                self.capture.release()
            self.camera_label.setText("📷 Camera đã tắt.")
            self.current_detections = [] # (ĐÃ THÊM) Xóa phát hiện cũ

    # --- UPDATE FRAME (ĐÃ SỬA LỖI 2, 3) ---
    def update_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame
        display_frame = frame.copy()

        # (ĐÃ SỬA LỖI 3) Reset danh sách phát hiện mỗi frame
        self.current_detections = []
        labels_to_draw = []

        # Detect pipeline: YOLO .pt -> crop -> CNN .h5 classify
        if self.realtime_enabled and self.model is not None and self.model_pt is not None:
            detections = self.run_yolo_on_frame(frame)
            for box in detections:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w * h < YOLO_MIN_AREA:
                    continue

                pad_w = int(w * YOLO_PADDING)
                pad_h = int(h * YOLO_PADDING)
                sx = max(0, x1 - pad_w)
                sy = max(0, y1 - pad_h)
                ex = min(frame.shape[1], x2 + pad_w)
                ey = min(frame.shape[0], y2 + pad_h)
                crop_bgr = frame[sy:ey, sx:ex]
                if crop_bgr.size == 0:
                    continue

                # (ĐÃ SỬA LỖI 2: BGR/RGB)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                # (ĐÃ SỬA LỖI 1: Kích thước)
                crop_resized = cv2.resize(crop_rgb, (IMG_WIDTH, IMG_HEIGHT)) 
                
                img = tf.keras.utils.img_to_array(crop_resized)
                img = np.expand_dims(img, 0) / 255.0
                preds = self.model.predict(img, verbose=0)[0]
                conf = float(np.max(preds))
                cls = int(np.argmax(preds))

                if conf < CNN_CONF_THRESHOLD:
                    continue

                name = CLASS_NAMES[cls]
                label = f"{DISPLAY_NAMES.get(name,name)} ({conf*100:.1f}%)"
                
                # (ĐÃ SỬA LỖI 3) Chỉ lưu lại để vẽ, không thêm vào bill
                labels_to_draw.append((label, (x1, y1, x2, y2)))
                self.current_detections.append((name, conf, (x1, y1, x2, y2)))

        # Fallback: Contours
        elif self.realtime_enabled and self.model is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < 2500:
                    continue
                crop_bgr = frame[y:y + h, x:x + w]
                
                # (ĐÃ SỬA LỖI 2: BGR/RGB)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                # (ĐÃ SỬA LỖI 1: Kích thước)
                crop_resized = cv2.resize(crop_rgb, (IMG_WIDTH, IMG_HEIGHT))
                
                img = tf.keras.utils.img_to_array(crop_resized)
                img = np.expand_dims(img, 0) / 255.0
                preds = self.model.predict(img, verbose=0)[0]
                conf = float(np.max(preds))
                cls = int(np.argmax(preds))
                if conf < CNN_CONF_THRESHOLD:
                    continue
                
                name = CLASS_NAMES[cls]
                label = f"{DISPLAY_NAMES[name]} ({conf * 100:.1f}%)"

                # (ĐÃ SỬA LỖI 3) Chỉ lưu lại để vẽ, không thêm vào bill
                labels_to_draw.append((label, (x, y, x + w, y + h)))
                self.current_detections.append((name, conf, (x, y, x + w, y + h)))
        
        # (ĐÃ SỬA LỖI 3) Vẽ tất cả các box sau khi đã phát hiện
        for (label, (x1, y1, x2, y2)) in labels_to_draw:
            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        # DISPLAY
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # --- RUN YOLO ON A FRAME ---
    # (Không thay đổi)
    def run_yolo_on_frame(self, frame):
        boxes = []
        if self.model_pt is None:
            return boxes
        try:
            if ULTRAYOLO is not None and isinstance(self.model_pt, ULTRAYOLO):
                results = self.model_pt.predict(frame, imgsz=640, conf=0.15, verbose=False)
                r = results[0]
                if hasattr(r, 'boxes') and r.boxes is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
                    if xyxy is not None:
                        for b in xyxy:
                            x1, y1, x2, y2 = map(int, b[:4])
                            boxes.append((x1, y1, x2, y2))
            else:
                results = self.model_pt(frame)
                try:
                    res = results.xyxy[0].cpu().numpy()
                except Exception:
                    try:
                        res = results[0].boxes.xyxy.cpu().numpy()
                    except Exception:
                        res = np.array([])
                for b in res:
                    x1, y1, x2, y2 = map(int, b[:4])
                    boxes.append((x1, y1, x2, y2))
        except Exception as e:
            print("⚠️ Lỗi khi chạy YOLO trên frame:", e)
        return boxes

    # --- ẢNH TĨNH ---
    # (Không thay đổi)
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image_path = file_name
            self.current_frame = None
            pixmap = QPixmap(file_name)
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            # (ĐÃ SỬA) Tắt realtime nếu đang bật
            if self.realtime_enabled:
                self.toggle_realtime()

    # (ĐÃ THÊM LỖI 3) Hàm kích hoạt lại nút
    def enable_detect_button(self):
        if hasattr(self, 'btn_detect'):
            self.btn_detect.setEnabled(True)
            print("ℹ️ Nút 'THÊM BÁNH' đã sẵn sàng.")

    # --- NHẬN DIỆN ẢNH (ĐÃ SỬA LỖI 2, 3) ---
    def run_detection(self):
        
        # (ĐÃ SỬA LỖI 3) Vô hiệu hóa nút
        self.btn_detect.setEnabled(False)
        QTimer.singleShot(DEBOUNCE_BUTTON_MS, self.enable_detect_button)
        
        if not self.model:
            print("⚠️ Model CNN chưa tải!")
            return

        now = time.time()
        
        # (ĐÃ SỬA LỖI 3) Logic mới: ưu tiên realtime
        # TRƯỜNG HỢP 1: Camera đang bật
        if self.realtime_enabled:
            if not self.current_detections:
                print("ℹ️ Camera đang bật nhưng chưa thấy bánh nào.")
                return
            
            print(f"✅ Thêm các bánh từ camera realtime:")
            item_added = False
            for (name, confidence, box) in self.current_detections:
                # Áp dụng cơ chế chống spam 2 giây cho TỪNG LOẠI BÁNH
                if name not in self.last_detected_time or now - self.last_detected_time[name] > 2:
                    print(f"  -> {name}: {confidence*100:.2f}%")
                    self.add_item_to_bill(name, PRICE_LIST.get(name, 0))
                    self.last_detected_time[name] = now # Lưu thời gian thêm
                    item_added = True
                else:
                    print(f"  -> (Bỏ qua {name}, mới thêm lúc {self.last_detected_time[name]:.0f})")
            if not item_added:
                 print("ℹ️ Tất cả bánh đều mới được thêm. Chờ 2 giây...")
            return # Dừng ở đây

        # TRƯỜNG HỢP 2: Dùng ảnh tĩnh (camera tắt)
        elif self.current_image_path:
            original = cv2.imread(self.current_image_path)
            if original is None:
                print("⚠️ Không đọc được ảnh:", self.current_image_path)
                return
        else:
            print("⚠️ Không có ảnh hoặc frame!")
            return

        # Code bên dưới chỉ chạy cho TRƯỜNG HỢP 2 (Ảnh tĩnh)
        display_frame = original.copy()

        # Pipeline 1: YOLO .pt
        if self.model_pt is not None:
            boxes = self.run_yolo_on_frame(original)
            for (x1, y1, x2, y2) in boxes:
                w, h = x2 - x1, y2 - y1
                if w * h < YOLO_MIN_AREA:
                    continue
                pad_w = int(w * YOLO_PADDING)
                pad_h = int(h * YOLO_PADDING)
                sx = max(0, x1 - pad_w)
                sy = max(0, y1 - pad_h)
                ex = min(original.shape[1], x2 + pad_w)
                ey = min(original.shape[0], y2 + pad_h)
                crop_bgr = original[sy:ey, sx:ex]
                if crop_bgr.size == 0:
                    continue

                # (ĐÃ SỬA LỖI 2: BGR/RGB)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                # (ĐÃ SỬA LỖI 1: Kích thước)
                crop_resized = cv2.resize(crop_rgb, (IMG_WIDTH, IMG_HEIGHT))
                
                img = tf.keras.utils.img_to_array(crop_resized)
                img = np.expand_dims(img, 0) / 255.0
                preds = self.model.predict(img, verbose=0)[0]
                conf = float(np.max(preds))
                cls = int(np.argmax(preds))
                if conf < CNN_CONF_THRESHOLD:
                    continue
                name = CLASS_NAMES[cls]
                label = f"{DISPLAY_NAMES.get(name,name)} ({conf*100:.1f}%)"
                color = (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # (ĐÃ SỬA LỖI 3) Với ảnh tĩnh, không cần debounce
                self.add_item_to_bill(name, PRICE_LIST.get(name, 0))

        # Pipeline 2: Fallback (Contours)
        else:
            # (Đoạn này đã đúng BGR->RGB, giữ nguyên)
            img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, 0) / 255.0
            preds = self.model.predict(img_array, verbose=0)[0]
            conf = float(np.max(preds))
            cls = int(np.argmax(preds))
            if conf < CNN_CONF_THRESHOLD:
                print("😕 Không phát hiện được bánh.")
            else:
                name = CLASS_NAMES[cls]
                print(f"✅ {name}: {conf*100:.2f}%")
                # (ĐÃ SỬA LỖI 3) Với ảnh tĩnh, không cần debounce
                self.add_item_to_bill(name, PRICE_LIST.get(name, 0))
                h, w = original.shape[:2]
                cv2.putText(display_frame, f"{DISPLAY_NAMES.get(name,name)} ({conf*100:.1f}%)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # Hiển thị ảnh tĩnh đã nhận diện
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # --- BILL ---
    # (Không thay đổi)
    def add_item_to_bill(self, cls, price):
        if cls in self.bill_items:
            self.bill_items[cls].increase_quantity()
        else:
            item = BillItemWidget(cls, price, self)
            self.bill_layout.addWidget(item)
            self.bill_items[cls] = item
        self.update_total()

    def update_total(self):
        total = sum(i.price * i.quantity for i in self.bill_items.values())
        self.subtotal_label.setText(f"Tổng tiền hàng:\n{total:,.0f}đ")
        self.final_total_label.setText(f"{total:,.0f}đ")

    # (ĐÃ SỬA LỖI 3) Kích hoạt lại nút khi xóa
    def clear_bill(self):
        for i in reversed(range(self.bill_layout.count())):
            w = self.bill_layout.takeAt(i).widget()
            if w: w.deleteLater()
        self.bill_items.clear()
        self.update_total()
        self.last_detected_time.clear() # Xóa lịch sử debounce của item
        self.enable_detect_button() # Kích hoạt lại nút
        print("🗑 Bill đã được xóa.")

    # (ĐÃ SỬA LỖI 3) Kích hoạt lại nút khi thanh toán
    def pay_bill(self):
        if not self.bill_items:
            print("⚠️ Bill trống!")
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = sum(i.price * i.quantity for i in self.bill_items.values())
        file_exists = os.path.exists("lich_su.csv")
        with open("lich_su.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Thời gian", "Tên bánh", "Số lượng", "Thành tiền", "Tổng hóa đơn"])
            for name, item in self.bill_items.items():
                if item.quantity > 0:
                    writer.writerow([now, name, item.quantity, item.price * item.quantity, total])
        print(f"💰 Thanh toán {total:,}đ — đã lưu lịch sử.")
        self.clear_bill() # Hàm này đã bao gồm cả việc kích hoạt lại nút

    def closeEvent(self, e):
        self.timer.stop()
        if self.capture: self.capture.release()
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BakeryApp()
    w.show()
    sys.exit(app.exec())



