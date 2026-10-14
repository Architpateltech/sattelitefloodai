from models.segmentation import DisasterSegmentation
from utils.image_utils import create_overlay
import cv2
from utils.database import Database
from PySide6.QtWidgets import QProgressBar
from utils.image_utils import (
    create_overlay,
    create_side_by_side
)
from utils.area_calculator import AreaCalculator
from utils.severity import get_severity
from models.predictor import DisasterPredictor
from PySide6.QtWidgets import QFileDialog
from PySide6.QtGui import QPixmap
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
    QTableWidgetItem
)

from PySide6.QtCore import Qt

from gui.styles import STYLE


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Satellite Disaster Intelligence System"
        )

        self.resize(1400, 850)

        self.setStyleSheet(STYLE)
        self.image_path = None
        self.predictor = DisasterPredictor()
        self.segmentor = DisasterSegmentation()
        self.area_calculator = AreaCalculator()
        self.database = Database()

        # Main Container
        main_widget = QWidget()

        self.setCentralWidget(main_widget)


        main_layout = QHBoxLayout()

        main_widget.setLayout(main_layout)


        # ======================
        # LEFT SIDEBAR
        # ======================

        sidebar = QFrame()

        sidebar.setFixedWidth(220)

        sidebar_layout = QVBoxLayout()


        logo = QLabel(
            "🛰\nSatellite AI"
        )

        logo.setAlignment(Qt.AlignCenter)

        logo.setStyleSheet(
            """
            font-size:24px;
            font-weight:bold;
            """
        )


        upload_btn = QPushButton(
            "📂 Upload Image"
        )
        upload_btn.clicked.connect(
        self.upload_image
        )


        analyze_btn = QPushButton(
            "🔍 Analyze"
        )
        analyze_btn.clicked.connect(
        self.analyze_image
        )

        history_btn = QPushButton(
            "📊 History"
        )
        history_btn.clicked.connect(
        self.show_history
        )

        settings_btn = QPushButton(
            "⚙ Settings"
        )
        analytics_btn = QPushButton(
        "📈 Analytics"
        )
        analytics_btn.clicked.connect(
        self.show_analytics
)
        report_btn = QPushButton(
            "📄 Generate Report"
        )
        report_btn.clicked.connect(
        self.generate_report
        )
        sidebar_layout.addWidget(logo)

        sidebar_layout.addSpacing(40)

        sidebar_layout.addWidget(upload_btn)

        sidebar_layout.addWidget(analyze_btn)

        sidebar_layout.addWidget(history_btn)

        sidebar_layout.addWidget(settings_btn)
        sidebar_layout.addWidget(report_btn)

        sidebar_layout.addStretch()
        sidebar_layout.addWidget (analytics_btn)

        sidebar.setLayout(sidebar_layout)



        # ======================
        # IMAGE VIEWER
        # ======================

        image_frame = QFrame()

        image_layout = QVBoxLayout()


        image_title = QLabel(
            "Satellite Image Preview"
        )

        image_title.setAlignment(
            Qt.AlignCenter
        )


        self.image_display = QLabel(
            "No Image Loaded"
        )

        self.image_display.setAlignment(
            Qt.AlignCenter
        )


        self.image_display.setStyleSheet(
            """
            border:2px dashed gray;
            font-size:20px;
            """
        )


        image_layout.addWidget(image_title)

        image_layout.addWidget(
            self.image_display
        )


        image_frame.setLayout(
            image_layout
        )



        # ======================
        # RESULT PANEL
        # ======================
        self.progress = QProgressBar()

        self.progress.setValue(
                0
        )
        result_layout.addWidget(
        self.progress
        )
        result_frame = QFrame()

        result_layout = QVBoxLayout()


        result_title = QLabel(
            "AI Analysis Result"
        )

        result_title.setStyleSheet(
            """
            font-size:20px;
            font-weight:bold;
            """
        )


        self.result_box = QTextEdit()

        self.result_box.setText(
            """
Disaster:
Waiting...

Confidence:
--

Affected Area:
--

Severity:
--
            """
        )


        self.result_box.setReadOnly(True)


        result_layout.addWidget(
            result_title
        )

        result_layout.addWidget(
            self.result_box
        )


        result_frame.setLayout(
            result_layout
        )


        # Add everything

        main_layout.addWidget(
            sidebar
        )

        main_layout.addWidget(
            image_frame,
            4
        )

        main_layout.addWidget(
            result_frame,
            2
        )
    def upload_image(self):

     file_path, _ = QFileDialog.getOpenFileName(
        self,
        "Select Satellite Image",
        "",
        "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
      )


     if file_path:

        self.image_path = file_path


        pixmap = QPixmap(file_path)


        self.image_display.setPixmap(
            pixmap.scaled(
                700,
                600,
                Qt.KeepAspectRatio
            )
        )


        self.result_box.setText(
            f"""
Image Loaded Successfully

File:
{file_path}

Status:
Ready for AI Analysis
            """
        )
    def analyze_image(self):
     self.progress.setValue(20)

     if self.image_path is None:

        self.result_box.setText(
            "Please upload an image first."
        )

        return


    # AI Prediction

    result = self.predictor.predict(
        self.image_path
    )

    self.progress.setValue(50)
    # Create segmentation mask

    mask = self.segmentor.create_mask(
        self.image_path
    )
    self.progress.setValue(75)
    area = self.area_calculator.calculate(
    mask
    )
    self.progress.setValue(100)

    severity = get_severity(
    area
    )
    self.database.save_analysis(
    result["disaster"],
    result["confidence"],
    area,
    severity
    )
    # Create overlay

    overlay = create_overlay(
        self.image_path,
        mask
    )


    # Save temporary result

    output_path = "outputs/analysis_result.png"


    comparison = create_side_by_side(
    self.image_path,
    overlay
)


    cv2.imwrite(
    output_path,
    comparison
)

    # Display overlay

    pixmap = QPixmap(
        output_path
    )


    self.image_display.setPixmap(
        pixmap.scaled(
            700,
            600,
            Qt.KeepAspectRatio
        )
    )


    # Display result

    self.result_box.setText(
    f"""
Disaster Type:
{result['disaster']}


Confidence:
{result['confidence']}%


Affected Area:
{area} km²


Severity:
{severity}


Segmentation:
Completed
"""
)
    def generate_report(self):

     from utils.report_generator import ReportGenerator


     result = self.predictor.predict(
        self.image_path
     )


     generator = ReportGenerator()


     generator.generate(
        "reports/disaster_report.pdf",
        result,
        self.image_path
    )


     self.result_box.setText(
        "PDF Report Generated Successfully"
    )
    def show_history(self):


        history = self.database.get_history()


        self.history_window = QWidget()

        self.history_window.setWindowTitle(
        "Analysis History"
    )


        self.history_window.resize(
            700,
            400
    )


        layout = QVBoxLayout()


        table = QTableWidget()


        table.setRowCount(
        len(history)
    )


        table.setColumnCount(
        6
    )


        table.setHorizontalHeaderLabels(
        [
            "ID",
            "Date",
            "Disaster",
            "Confidence",
            "Area",
            "Severity"
        ]
    )



        for row, data in enumerate(history):

          for col, value in enumerate(data):

             table.setItem(
                row,
                col,
                QTableWidgetItem(
                    str(value)
                )
            )


        layout.addWidget(
        table
    )


        self.history_window.setLayout(
        layout
    )


        self.history_window.show()
    def show_analytics(self):


     from utils.analytics import Analytics


     analytics = Analytics(
        self.database
    )


     data = analytics.get_statistics()


     self.result_box.setText(

        f"""
        📊 SYSTEM ANALYTICS


Total Analyses:
{data['cases']}


Total Affected Area:
{data['area']} km²


Disaster Distribution:

{data['disasters']}
        """

    )