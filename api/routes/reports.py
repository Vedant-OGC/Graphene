"""
Graphene — Report Routes

Endpoints for generating and downloading FIU Suspicious Transaction
Reports (STR) as PDF documents.
"""

import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import AppState, get_app_state
from api.models.responses import (
    ReportGenerateRequest,
    ReportGenerateResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "REPORTS_DIR": str(
        Path(__file__).resolve().parent.parent.parent / "reports"
    ),
}

router = APIRouter()


def _generate_str_pdf(
    alert: dict,
    account_metadata: dict,
    transactions: list[dict],
    analyst_name: str,
    analyst_id: str,
    report_id: str,
) -> bytes:
    """Generate a Suspicious Transaction Report PDF.

    Formatted for FIU-IND with full evidence, SHAP explanation,
    and transaction details.

    Args:
        alert: Alert data dictionary.
        account_metadata: Account metadata from Neo4j.
        transactions: Recent transactions for the account.
        analyst_name: Name of the reporting analyst.
        analyst_id: ID of the reporting analyst.
        report_id: UUID for this report.

    Returns:
        PDF content as bytes.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "GrapheneTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#00E5CC"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "GrapheneHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#0A0E1A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "GrapheneBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "GrapheneSmall",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.gray,
    )

    elements = []

    elements.append(Paragraph("GRAPHENE", title_style))
    elements.append(
        Paragraph(
            "Fund Flow Intelligence — Suspicious Transaction Report",
            body_style,
        )
    )
    elements.append(Spacer(1, 8 * mm))

    elements.append(
        Paragraph("Reporting Entity", heading_style)
    )
    entity_data = [
        ["Institution:", "Union Bank of India"],
        ["Report Type:", "Suspicious Transaction Report (STR)"],
        ["FIU Report No:", report_id[:16].upper()],
        ["Date:", datetime.now().strftime("%d-%m-%Y")],
        ["Time (IST):", datetime.now().strftime("%H:%M:%S")],
    ]
    entity_table = Table(entity_data, colWidths=[4 * cm, 12 * cm])
    entity_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(entity_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(
        Paragraph("Subject Account Details", heading_style)
    )
    acct_data = [
        [
            "Account ID:",
            alert.get("account_id", "N/A"),
        ],
        [
            "Customer Name:",
            account_metadata.get("customer_name", "N/A"),
        ],
        [
            "Account Type:",
            account_metadata.get("account_type", "N/A"),
        ],
        [
            "Category:",
            account_metadata.get("customer_category", "N/A"),
        ],
        [
            "Branch:",
            account_metadata.get("branch_code", "N/A"),
        ],
        [
            "KYC Status:",
            account_metadata.get("kyc_status", "N/A"),
        ],
    ]
    acct_table = Table(acct_data, colWidths=[4 * cm, 12 * cm])
    acct_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(acct_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(
        Paragraph("Risk Assessment", heading_style)
    )
    risk_score = alert.get("risk_score", 0)
    risk_tier = alert.get("risk_tier", "LOW")
    tier_color = {
        "CRITICAL": "#FF4757",
        "HIGH": "#FF6B35",
        "MEDIUM": "#FFA502",
        "LOW": "#2ED573",
    }.get(risk_tier, "#636E72")

    elements.append(
        Paragraph(
            f'Risk Score: <b>{risk_score:.0f}/100</b> — '
            f'<font color="{tier_color}"><b>{risk_tier}</b></font>',
            body_style,
        )
    )
    triggered = alert.get("triggered_patterns", [])
    if triggered:
        elements.append(
            Paragraph(
                f"AML Patterns Triggered: <b>{', '.join(triggered)}</b>",
                body_style,
            )
        )
    elements.append(Spacer(1, 4 * mm))

    elements.append(
        Paragraph("Evidence and Explanation", heading_style)
    )
    evidence = alert.get("evidence_summary", "No evidence summary available.")
    elements.append(Paragraph(evidence, body_style))
    elements.append(Spacer(1, 6 * mm))

    if transactions:
        elements.append(
            Paragraph("Suspicious Transactions", heading_style)
        )
        txn_header = [
            "Txn ID", "Amount (₹)", "Type",
            "Timestamp", "Counterparty", "Fraud",
        ]
        txn_rows = [txn_header]
        for txn in transactions[:20]:
            txn_rows.append([
                str(txn.get("txn_id", ""))[:12] + "...",
                f"₹{txn.get('amount', 0):,.2f}",
                str(txn.get("txn_type", "")),
                str(txn.get("timestamp", ""))[:19],
                str(txn.get("counterparty", ""))[:13],
                "YES" if txn.get("is_fraud") else "NO",
            ])

        txn_table = Table(
            txn_rows,
            colWidths=[
                2.5 * cm, 2.5 * cm, 1.5 * cm,
                3.5 * cm, 3 * cm, 1.5 * cm,
            ],
        )
        txn_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A0E1A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor("#F8F9FA"),
            ]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(txn_table)
        elements.append(Spacer(1, 6 * mm))

    elements.append(
        Paragraph("Analyst Certification", heading_style)
    )
    elements.append(
        Paragraph(
            f"Analyst Name: <b>{analyst_name}</b>",
            body_style,
        )
    )
    elements.append(
        Paragraph(
            f"Analyst ID: <b>{analyst_id}</b>",
            body_style,
        )
    )
    elements.append(
        Paragraph(
            f"Date: <b>{datetime.now().strftime('%d-%m-%Y %H:%M IST')}</b>",
            body_style,
        )
    )
    elements.append(Spacer(1, 10 * mm))
    elements.append(
        Paragraph(
            "This report has been generated by Graphene — "
            "AI-powered Fund Flow Intelligence System. "
            "Confidential — Restricted to authorised personnel.",
            small_style,
        )
    )

    doc.build(elements)
    return buffer.getvalue()


@router.post(
    "/reports/generate",
    response_model=ReportGenerateResponse,
    tags=["Reports"],
    summary="Generate FIU Suspicious Transaction Report",
)
def generate_report(
    request: ReportGenerateRequest,
    state: AppState = Depends(get_app_state),
) -> ReportGenerateResponse:
    """Generate a PDF STR report for a specific alert.

    Creates a formatted PDF document with all evidence, SHAP
    explanation, and transaction details for filing with FIU-IND.
    """
    alert = state.get_alert_by_id(request.alert_id)
    if not alert:
        raise HTTPException(
            status_code=404,
            detail=f"Alert {request.alert_id} not found.",
        )

    account_id = alert.get("account_id", "")
    metadata = state.get_account_metadata(account_id)
    transactions = state.get_recent_transactions(account_id)

    report_id = str(uuid4())

    try:
        pdf_bytes = _generate_str_pdf(
            alert=alert,
            account_metadata=metadata,
            transactions=transactions,
            analyst_name=request.analyst_name,
            analyst_id=request.analyst_id,
            report_id=report_id,
        )

        os.makedirs(CONFIG["REPORTS_DIR"], exist_ok=True)
        pdf_path = os.path.join(
            CONFIG["REPORTS_DIR"], f"{report_id}.pdf"
        )
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        state.store_report(
            report_id,
            {
                "report_id": report_id,
                "alert_id": request.alert_id,
                "account_id": account_id,
                "pdf_path": pdf_path,
                "generated_at": datetime.now().isoformat(),
            },
        )

        logger.info(
            "Generated STR report %s for alert %s",
            report_id,
            request.alert_id,
        )

        return ReportGenerateResponse(
            report_id=report_id,
            download_url=f"/api/reports/{report_id}/download",
            generated_at=datetime.now().isoformat(),
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab is not installed. Cannot generate PDFs.",
        )
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {e}",
        )


@router.get(
    "/reports/{report_id}/download",
    tags=["Reports"],
    summary="Download a generated report",
)
def download_report(
    report_id: str,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    """Stream a previously generated PDF report.

    Returns the PDF as an application/pdf response for browser
    viewing or download.
    """
    report = state.get_report(report_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found.",
        )

    pdf_path = report.get("pdf_path", "")
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="Report PDF file not found on disk.",
        )

    def iterfile():
        with open(pdf_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="STR_{report_id[:8]}.pdf"'
            ),
        },
    )

# Updated: 2026-01-12
