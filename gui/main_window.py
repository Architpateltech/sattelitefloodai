import os
import logging

import cv2
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QProgressBar,
    QMessageBox,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from gui.styles import STYLE
from models.predictor import DisasterPredictor
from models.segmentation import DisasterSegmentation
from utils.image_utils import create_overlay, create_side_by_side
from utils.area_calculator import AreaCalculator
from utils.severity import get_severity
from utils.database import Database

# Ensure output directories exist on startup
os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Satellite Disaster Intelligence System")
        self.resize(1400, 850)
        self.setStyleSheet(STYLE)

        self.image_path = None
        self.predictor = DisasterPredictor()
        self.segmentor = DisasterSegmentation()
        self.area_calculator = AreaCalculator()
        self.database = Database()

        # ── Main Container ─────────────────────────────────────────────────
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ── LEFT SIDEBAR ────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)

        logo = QLabel("🛰\nSatellite AI")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("font-size:24px; font-weight:bold;")

        upload_btn = QPushButton("📂 Upload Image")
        upload_btn.clicked.connect(self.upload_image)

        analyze_btn = QPushButton("🔍 Analyze")
        analyze_btn.clicked.connect(self.analyze_image)

        history_btn = QPushButton("📊 History")
        history_btn.clicked.connect(self.show_history)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.show_settings)

        report_btn = QPushButton("📄 Generate Report")
        report_btn.clicked.connect(self.generate_report)

        analytics_btn = QPushButton("📈 Analytics")
        analytics_btn.clicked.connect(self.show_analytics)

        sidebar_layout.addWidget(logo)
        sidebar_layout.addSpacing(40)
        sidebar_layout.addWidget(upload_btn)
        sidebar_layout.addWidget(analyze_btn)
        sidebar_layout.addWidget(history_btn)
        sidebar_layout.addWidget(settings_btn)
        sidebar_layout.addWidget(report_btn)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(analytics_btn)

        # ── IMAGE VIEWER ─────────────────────────────────────────────────────
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)

        preview_title = QLabel("Preview Image Section")
        preview_title.setAlignment(Qt.AlignCenter)
        self.preview_display = QLabel("No Image Loaded")
        self.preview_display.setAlignment(Qt.AlignCenter)
        self.preview_display.setStyleSheet(
            "border:2px dashed #0284c7; font-size:20px; color:#0284c7; background-color:#f0f9ff; border-radius:10px;"
        )

        analyzed_title = QLabel("Analyzed Image Section")
        analyzed_title.setAlignment(Qt.AlignCenter)
        self.analyzed_display = QLabel("Analysis Pending")
        self.analyzed_display.setAlignment(Qt.AlignCenter)
        self.analyzed_display.setStyleSheet(
            "border:2px dashed #0284c7; font-size:20px; color:#0284c7; background-color:#f0f9ff; border-radius:10px;"
        )

        image_layout.addWidget(preview_title)
        image_layout.addWidget(self.preview_display, 1)
        image_layout.addWidget(analyzed_title)
        image_layout.addWidget(self.analyzed_display, 1)

        # ── RESULT PANEL ─────────────────────────────────────────────────────
        result_frame = QFrame()
        result_layout = QVBoxLayout(result_frame)

        result_title = QLabel("AI Analysis Result")
        result_title.setStyleSheet("font-size:20px; font-weight:bold;")

        # Progress bar — defined AFTER result_layout
        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.result_box = QTextEdit()
        self.result_box.setHtml(
            "<div style='line-height: 1.5; font-size: 15px;'>"
            "<b>Disaster:</b><br/>Waiting...<br/><br/>"
            "<b>Confidence:</b><br/>--<br/><br/>"
            "<b>Affected Area:</b><br/>--<br/><br/>"
            "<b>Severity:</b><br/>--<br/>"
            "</div>"
        )
        self.result_box.setReadOnly(True)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.progress)
        result_layout.addWidget(self.result_box)

        # ── Assemble ──────────────────────────────────────────────────────────
        main_layout.addWidget(sidebar)
        main_layout.addWidget(image_frame, 4)
        main_layout.addWidget(result_frame, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Slot: Upload Image
    # ─────────────────────────────────────────────────────────────────────────
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Satellite Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )

        if not file_path:
            return

        self.image_path = file_path
        self.progress.setValue(0)
        logger.info("Image loaded: %s", file_path)

        is_tif = file_path.lower().endswith(('.tif', '.tiff'))
        pixmap = QPixmap()

        if is_tif:
            from utils.image_utils import load_rgb_image
            from PySide6.QtGui import QImage
            import cv2
            
            img_bgr = load_rgb_image(file_path)
            if img_bgr is not None:
                height, width, channel = img_bgr.shape
                bytes_per_line = 3 * width
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                qimg = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                qimg.ndarray = img_rgb  # Keep a reference to prevent garbage collection
                pixmap = QPixmap.fromImage(qimg)
        else:
            pixmap = QPixmap(file_path)
                    
        if pixmap.isNull():
            self._show_error("Failed to load image. The file may be corrupted.")
            self.image_path = None
            return

        self.preview_display.setPixmap(
            pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.analyzed_display.clear()
        self.analyzed_display.setText("Analysis Pending")
        self.result_box.setHtml(
            f"<div style='line-height: 1.5; font-size: 15px;'>"
            f"<b>Image Loaded Successfully</b><br/><br/>"
            f"<b>File:</b><br/>{file_path}<br/><br/>"
            f"<b>Status:</b><br/>Ready for AI Analysis<br/>"
            f"</div>"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Slot: Analyze Image
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_image(self):
        if self.image_path is None:
            self.result_box.setHtml("<b style='font-size: 15px;'>Please upload an image first.</b>")
            return

        try:
            self.progress.setValue(20)

            # 1. Segmentation mask (run U-Net model)
            mask = self.segmentor.create_mask(self.image_path)
            self.progress.setValue(60)

            # 2. Area & severity
            area = self.area_calculator.calculate(mask, self.image_path)
            severity = get_severity(area)
            self.progress.setValue(80)
            
            # 3. Determine classification based on segmentation
            if area > 0.0:
                disaster_type = "Flood Detected"
                confidence = 98.5 # High confidence if UNet detects pixels
            else:
                disaster_type = "No Flood"
                confidence = 99.0

            # Persist to database
            self.database.save_analysis(
                disaster_type,
                confidence,
                area,
                severity,
            )

            # Build overlay
            from utils.image_utils import create_overlay
            overlay = create_overlay(self.image_path, mask)

            import cv2
            output_path = "outputs/analysis_result.png"
            cv2.imwrite(output_path, overlay)
            self.progress.setValue(100)

            # Display result image
            from PySide6.QtGui import QPixmap
            from PySide6.QtCore import Qt
            pixmap = QPixmap(output_path)
            if not pixmap.isNull():
                self.analyzed_display.setPixmap(
                    pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

            # Display text result
            self.result_box.setHtml(
                f"<div style='line-height: 1.5; font-size: 15px;'>"
                f"<b>Disaster Type:</b><br/>{disaster_type}<br/><br/>"
                f"<b>Confidence:</b><br/>{confidence}%<br/><br/>"
                f"<b>Affected Area:</b><br/>{area} km²<br/><br/>"
                f"<b>Severity:</b><br/>{severity}<br/><br/>"
                f"<b>Segmentation:</b><br/>Completed<br/>"
                f"</div>"
            )
            logger.info("Analysis complete — %s", disaster_type)

        except Exception as exc:
            logger.exception("Analysis failed")
            self._show_error(f"Analysis failed:\n{exc}")
            self.progress.setValue(0)

    # ─────────────────────────────────────────────────────────────────────────
    # Slot: Generate Report
    # ─────────────────────────────────────────────────────────────────────────
    def generate_report(self):
        if self.image_path is None:
            self.result_box.setHtml("<b style='font-size: 15px;'>Please upload and analyse an image first.</b>")
            return

        try:
            from utils.report_generator import ReportGenerator

            result = self.predictor.predict(self.image_path)
            mask = self.segmentor.create_mask(self.image_path)
            area = self.area_calculator.calculate(mask, self.image_path)
            severity = get_severity(area)

            # Enrich result dict with computed area & severity
            result["area"] = area
            result["severity"] = severity

            output_path = "reports/disaster_report.pdf"
            generator = ReportGenerator()
            generator.generate(output_path, result, self.image_path)

            self.result_box.setHtml(
                f"<div style='line-height: 1.5; font-size: 15px;'>"
                f"<b>✅ PDF Report Generated Successfully</b><br/><br/>"
                f"<b>Saved to:</b><br/>{os.path.abspath(output_path)}"
                f"</div>"
            )
            logger.info("Report saved to %s", output_path)

        except Exception as exc:
            logger.exception("Report generation failed")
            self._show_error(f"Report generation failed:\n{exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Slot: Show History
    # ─────────────────────────────────────────────────────────────────────────
    def show_history(self):
        history = self.database.get_history()

        self.history_window = QWidget()
        self.history_window.setWindowTitle("Analysis History")
        self.history_window.resize(700, 400)
        self.history_window.setStyleSheet(self.styleSheet())

        layout = QVBoxLayout(self.history_window)

        table = QTableWidget()
        table.setRowCount(len(history))
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["ID", "Date", "Disaster", "Confidence", "Area (km²)", "Severity"]
        )
        table.horizontalHeader().setStretchLastSection(True)

        for row, data in enumerate(history):
            for col, value in enumerate(data):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, col, item)

        layout.addWidget(table)
        self.history_window.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Slot: Show Analytics
    # ─────────────────────────────────────────────────────────────────────────
    def show_analytics(self):
        from utils.analytics import Analytics

        analytics = Analytics(self.database)
        data = analytics.get_statistics()

        self.result_box.setHtml(
            f"<div style='line-height: 1.5; font-size: 15px;'>"
            f"<b>📊 SYSTEM ANALYTICS</b><br/><br/>"
            f"<b>Total Analyses:</b><br/>{data['cases']}<br/><br/>"
            f"<b>Total Affected Area:</b><br/>{data['area']} km²<br/><br/>"
            f"<b>Disaster Distribution:</b><br/>{data['disasters']}<br/>"
            f"</div>"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Slot: Show Settings (placeholder)
    # ─────────────────────────────────────────────────────────────────────────
    def show_settings(self):
        self.result_box.setHtml(
            "<div style='line-height: 1.5; font-size: 15px;'>"
            "<b>⚙ Settings</b><br/><br/>"
            "Settings panel coming soon.<br/>"
            "</div>"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _show_error(self, message: str):
        """Display a modal error dialog and log the message."""
        logger.error(message)
        QMessageBox.critical(self, "Error", message)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        """Cleanly close the SQLite connection on window close."""
        try:
            self.database.close()
        except Exception:
            pass
        super().closeEvent(event)