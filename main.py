import sys, os, csv, time, numpy as np, tensorflow as tf, cv2
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QScrollArea
)
from PyQt6.QtGui import QPixmap, QFont, QImage
from PyQt6.QtCore import Qt, QTimer


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
IMG_WIDTH, IMG_HEIGHT = 180, 180
MODEL_PATH = "merged_multi_label_cnn.h5"


# --- ITEM TRONG BILL ---
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
        self.setGeometry(100, 100, 1200, 700)
        self.model = None
        self.bill_items, self.last_detected_time = {}, {}
        self.realtime_enabled, self.capture = False, None
        self.current_image_path, self.current_frame = None, None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.load_model()
        self.setup_ui()
        self.apply_style()

    # --- MODEL ---
    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded.")
        else:
            print("❌ Không tìm thấy model!")

    # --- UI ---
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
        btn_detect = QPushButton("📸 NHẬN DIỆN ẢNH")
        btn_detect.clicked.connect(self.run_detection)
        btn_pay = QPushButton("💰 THANH TOÁN")
        btn_pay.clicked.connect(self.pay_bill)

        c1.addWidget(logo)
        c1.addWidget(self.camera_label)
        c1.addWidget(self.btn_realtime)
        c1.addWidget(btn_load)
        c1.addWidget(btn_detect)
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
        """)

    # --- CAMERA ---
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

    # --- CẬP NHẬT FRAME ---
    def update_frame(self):
        if not self.capture:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame
        display_frame = frame.copy()

        # --- DETECT REALTIME ---
        if self.realtime_enabled and self.model is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            now = time.time()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < 2500:
                    continue
                crop = frame[y:y + h, x:x + w]
                crop_resized = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))
                img = tf.keras.utils.img_to_array(crop_resized)
                img = tf.expand_dims(img, 0)
                preds = self.model.predict(img, verbose=0)[0]
                conf = np.max(preds)
                cls = np.argmax(preds)
                if conf < 0.6:
                    continue
                name = CLASS_NAMES[cls]
                label = f"{DISPLAY_NAMES[name]} ({conf * 100:.1f}%)"
                color = (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # tránh trùng
                if name not in self.last_detected_time or now - self.last_detected_time[name] > 2:
                    self.add_item_to_bill(name, PRICE_LIST.get(name, 0))
                    self.last_detected_time[name] = now

        # HIỂN THỊ
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # --- ẢNH TĨNH ---
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

    # --- NHẬN DIỆN ẢNH ---
    def run_detection(self):
        if not self.model:
            print("⚠️ Model chưa tải!")
            return
        img_array = None
        if self.current_frame is not None:
            img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
        elif self.current_image_path:
            img = tf.keras.utils.load_img(self.current_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
        else:
            print("⚠️ Không có ảnh hoặc frame!")
            return

        preds = self.model.predict(img_array, verbose=0)[0]
        conf = np.max(preds)
        cls = np.argmax(preds)
        if conf < 0.6:
            print("😕 Không phát hiện được bánh.")
            return
        name = CLASS_NAMES[cls]
        print(f"✅ {name}: {conf*100:.2f}%")
        self.add_item_to_bill(name, PRICE_LIST.get(name, 0))

    # --- BILL ---
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

    def clear_bill(self):
        for i in reversed(range(self.bill_layout.count())):
            w = self.bill_layout.takeAt(i).widget()
            if w: w.deleteLater()
        self.bill_items.clear()
        self.update_total()

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
                writer.writerow([now, name, item.quantity, item.price * item.quantity, total])
        print(f"💰 Thanh toán {total:,}đ — đã lưu lịch sử.")
        self.clear_bill()

    def closeEvent(self, e):
        self.timer.stop()
        if self.capture: self.capture.release()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BakeryApp()
    w.show()
    sys.exit(app.exec())
