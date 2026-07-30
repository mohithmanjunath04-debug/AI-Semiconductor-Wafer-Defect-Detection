from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile


def generate_pdf(
    wafer_no,
    actual_label,
    predicted_label,
    confidence,
    accuracy,
    precision,
    recall,
    f1,
    top3
):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    pdf = canvas.Canvas(
        temp_file.name,
        pagesize=letter
    )

    y = 760

    pdf.setFont("Helvetica-Bold",16)
    pdf.drawString(
        60,
        y,
        "AI Semiconductor Wafer Defect Report"
    )

    y -= 40

    pdf.setFont("Helvetica",11)

    pdf.drawString(60,y,f"Wafer Number : {wafer_no}")
    y -= 25

    pdf.drawString(60,y,f"Actual Label : {actual_label}")
    y -= 25

    pdf.drawString(60,y,f"Predicted Label : {predicted_label}")
    y -= 25

    pdf.drawString(
        60,
        y,
        f"Confidence : {confidence:.2f}%"
    )

    y -= 40

    pdf.drawString(
        60,
        y,
        f"Accuracy : {accuracy:.2f}%"
    )

    y -= 25

    pdf.drawString(
        60,
        y,
        f"Precision : {precision:.2f}%"
    )

    y -= 25

    pdf.drawString(
        60,
        y,
        f"Recall : {recall:.2f}%"
    )

    y -= 25

    pdf.drawString(
        60,
        y,
        f"F1 Score : {f1:.2f}%"
    )

    y -= 45

    pdf.setFont("Helvetica-Bold",13)
    pdf.drawString(
        60,
        y,
        "Top 3 Predictions"
    )

    pdf.setFont("Helvetica",11)

    y -= 25

    for i,row in top3.iterrows():

        pdf.drawString(
            80,
            y,
            f"{row['Defect']} : {row['Probability (%)']:.2f}%"
        )

        y -= 20

    y -= 30

    pdf.drawString(
        60,
        y,
        "Generated using AI Semiconductor Wafer Defect Detection"
    )

    y -= 20

    pdf.drawString(
        60,
        y,
        "Developer : Mohith Manjunath"
    )

    pdf.save()

    return temp_file.name