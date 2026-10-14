import logging
import os
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

logger = logging.getLogger(__name__)


class ReportGenerator:

    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate(
        self,
        output_path: str,
        result: dict,
        image_path: Optional[str] = None,
    ):
        """
        Generate a PDF disaster-analysis report.

        Args:
            output_path: Destination path for the PDF file.
            result:      Dict with keys 'disaster', 'confidence', 'area', 'severity'.
            image_path:  Optional path to the satellite image to embed.
        """
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        content = []

        # Title
        content.append(
            Paragraph("Satellite Disaster Analysis Report", self.styles["Title"])
        )
        content.append(Spacer(1, 20))

        # Body text
        disaster   = result.get("disaster", "N/A")
        confidence = result.get("confidence", "N/A")
        area       = result.get("area", "N/A")
        severity   = result.get("severity", "N/A")

        data = (
            f"<b>Disaster Type:</b> {disaster}<br/>"
            f"<b>Confidence:</b> {confidence}%<br/>"
            f"<b>Affected Area:</b> {area} km²<br/>"
            f"<b>Severity:</b> {severity}<br/>"
        )
        content.append(Paragraph(data, self.styles["Normal"]))
        content.append(Spacer(1, 20))

        # Embed image if provided and readable
        if image_path and os.path.exists(image_path):
            try:
                img = Image(image_path, width=400, height=300)
                content.append(img)
            except Exception as exc:
                logger.warning("Could not embed image in report: %s", exc)

        doc.build(content)
        logger.info("Report saved to '%s'", output_path)