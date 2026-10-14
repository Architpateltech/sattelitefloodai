from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet



class ReportGenerator:


    def __init__(self):

        self.styles = getSampleStyleSheet()



    def generate(
            self,
            output_path,
            result,
            image_path=None
    ):


        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter
        )


        content = []


        title = Paragraph(
            "Satellite Disaster Analysis Report",
            self.styles["Title"]
        )


        content.append(title)

        content.append(
            Spacer(1,20)
        )


        data = f"""

Disaster Type:
{result['disaster']}


Confidence:
{result['confidence']}%


Affected Area:
{result['area']} km²


Severity:
{result['severity']}


"""


        report_text = Paragraph(
            data.replace(
                "\n",
                "<br/>"
            ),
            self.styles["Normal"]
        )


        content.append(
            report_text
        )


        content.append(
            Spacer(1,20)
        )


        if image_path:

            img = Image(
                image_path,
                width=400,
                height=300
            )

            content.append(
                img
            )


        doc.build(
            content
        )


        print(
            "Report Generated:",
            output_path
        )