import streamlit as st
import fitz
import json
import pandas as pd
import plotly.express as px
import docx
import io
import os
import logging

from google import genai
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Invoice Operations & Compliance Platform",
    layout="wide"
)

st.title("🛡️ AI Invoice Operations & Compliance Platform")

st.caption(
    "AI-powered invoice processing, GST validation, compliance automation, "
    "financial risk analysis, and enterprise invoice operations."
)

# =========================================================
# FOLDERS
# =========================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    filename="logs/system.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# DATABASE
# =========================================================

DATABASE_URL = "sqlite:///invoice_platform.db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

# =========================================================
# DATABASE MODEL
# =========================================================

class InvoiceRecord(Base):
    __tablename__ = "invoice_records"

    id = Column(Integer, primary_key=True)

    filename = Column(String)
    vendor_name = Column(String)
    invoice_number = Column(String)
    invoice_date = Column(String)

    currency = Column(String)

    subtotal = Column(Float)
    tax = Column(Float)
    total = Column(Float)

    confidence = Column(Float)
    fraud_score = Column(Float)

    status = Column(String)
    risks = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Workspace")

    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ AI Engine Connected")

    except Exception:
        gemini_api_key = None
        st.error("❌ Gemini API Key Missing")

    st.info("Get your key from Google AI Studio.")

    st.divider()

    st.subheader("👑 Account Tier")

    user_tier = st.selectbox(
        "Select Tier",
        [
            "Free (10 Files)",
            "Pro (1,000 Files)",
            "Enterprise (Unlimited)"
        ]
    )

    if user_tier == "Free (10 Files)":
        upload_limit = 10

    elif user_tier == "Pro (1,000 Files)":
        upload_limit = 1000

    else:
        upload_limit = 10000

    st.success("Enterprise Audit Engine Online")

# =========================================================
# PYDANTIC SCHEMA
# =========================================================

class LineItem(BaseModel):
    item: str
    qty: int
    unit_price: float

class InvoiceSchema(BaseModel):
    vendor_name: str
    invoice_date: str
    invoice_number: str

    currency: str

    subtotal: float
    vat_or_tax_amount: float
    total_amount: float

    confidence_score: float

    line_items: list[LineItem]

# =========================================================
# FILE UTILITIES
# =========================================================

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def extract_content_smart(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    # PDF
    if file_type == "pdf":
        doc = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        text = ""

        for page in doc:
            text += page.get_text()

        # scanned PDF detection
        if len(text.strip()) < 100:
            pix = doc[0].get_pixmap()

            img_data = pix.tobytes("png")

            return {
                "type": "image",
                "data": img_data,
                "mime_type": "image/png"
            }

        return {
            "type": "text",
            "data": text
        }

    # DOCX
    elif file_type in ["docx", "doc"]:
        document = docx.Document(uploaded_file)

        full_text = []

        for para in document.paragraphs:
            full_text.append(para.text)

        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text for cell in row.cells
                )

                full_text.append(row_text)

        return {
            "type": "text",
            "data": "\n".join(full_text)
        }

    # IMAGE
    elif file_type in ["jpg", "jpeg", "png"]:
        return {
            "type": "image",
            "data": uploaded_file.getvalue(),
            "mime_type": f"image/{file_type}"
        }

    return None

# =========================================================
# AI EXTRACTION ENGINE
# =========================================================

def process_invoice_agent(api_key, content_payload):
    client = genai.Client(api_key=api_key)

    system_prompt = """
    You are an enterprise-grade AI Invoice Operations Agent.

    RULES:
    1. Extract values EXACTLY as written.
    2. Never correct mathematical mistakes.
    3. Return STRICT JSON.
    4. Lower confidence if OCR quality is weak.
    5. Detect suspicious inconsistencies.
    """

    contents = [system_prompt]

    if content_payload["type"] == "text":
        contents.append(
            f"Analyze this invoice:\n{content_payload['data']}"
        )

    else:
        contents.append("Analyze this invoice image.")

        contents.append({
            "data": content_payload["data"],
            "mime_type": content_payload["mime_type"]
        })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_schema": InvoiceSchema,
            "temperature": 0.1
        }
    )

    return json.loads(response.text)

# =========================================================
# FRAUD ENGINE
# =========================================================

BLACKLISTED_VENDORS = [
    "Fake Corp",
    "Shell Company Ltd",
    "Unknown Vendor"
]

class FraudDetector:
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

    def anomaly_score(
        self,
        historical_totals,
        current_total
    ):
        if len(historical_totals) < 5:
            return 0.1

        df = pd.DataFrame({
            "amount": historical_totals
        })

        self.model.fit(df)

        prediction = self.model.predict(
            [[current_total]]
        )

        if prediction[0] == -1:
            return 0.95

        return 0.15

def run_compliance_audit(
    data,
    fraud_history,
    historical_totals
):
    flags = []

    # DUPLICATE CHECK
    invoice_id = (
        f"{data['vendor_name']}|"
        f"{data['invoice_number']}|"
        f"{data['total_amount']}"
    )

    if invoice_id in fraud_history:
        flags.append(
            f"Duplicate invoice detected ({data['invoice_number']})"
        )

    else:
        fraud_history.add(invoice_id)

    # MATH VALIDATION
    calc_subtotal = sum(
        item["qty"] * item["unit_price"]
        for item in data["line_items"]
    )

    if abs(calc_subtotal - data["subtotal"]) > 1:
        flags.append(
            "Line item subtotal mismatch detected"
        )

    calc_total = (
        data["subtotal"] +
        data["vat_or_tax_amount"]
    )

    if abs(calc_total - data["total_amount"]) > 1:
        flags.append(
            "Invoice total mismatch detected"
        )

    # GST VALIDATION
    if data["currency"] == "INR":
        expected_tax = data["subtotal"] * 0.18

        if abs(expected_tax - data["vat_or_tax_amount"]) > 2:
            flags.append(
                "Potential GST compliance issue"
            )

    # VAT VALIDATION
    elif data["currency"] == "GBP":
        expected_tax = data["subtotal"] * 0.20

        if abs(expected_tax - data["vat_or_tax_amount"]) > 2:
            flags.append(
                "Potential VAT compliance issue"
            )

    # BLACKLIST CHECK
    if any(
        bad.lower() in data["vendor_name"].lower()
        for bad in BLACKLISTED_VENDORS
    ):
        flags.append(
            "Vendor appears in high-risk watchlist"
        )

    # CONFIDENCE CHECK
    if data["confidence_score"] < 0.85:
        flags.append(
            "Low confidence extraction - human review recommended"
        )

    # ML ANOMALY DETECTION
    detector = FraudDetector()

    fraud_score = detector.anomaly_score(
        historical_totals,
        data["total_amount"]
    )

    if fraud_score > 0.9:
        flags.append(
            "Anomalous invoice amount detected"
        )

    status = "✅ Verified"

    if flags:
        status = "⚠️ Flagged"

    return status, flags, fraud_score

# =========================================================
# SESSION STATE
# =========================================================

if "fraud_history" not in st.session_state:
    st.session_state["fraud_history"] = set()

if "historical_totals" not in st.session_state:
    st.session_state["historical_totals"] = []

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📤 Invoice Processing & Compliance",
    "📊 Analytics Dashboard",
    "🗄️ Invoice Database"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:
    st.subheader("📤 Upload Source Documents")

    uploaded_files = st.file_uploader(
        f"Supported: PDF, PNG, JPG, DOCX (Limit: {upload_limit})",
        type=[
            "pdf",
            "docx",
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True
    )

    if uploaded_files and gemini_api_key:
        if len(uploaded_files) > upload_limit:
            st.error("Tier limit exceeded.")

        else:
            if st.button(
                "🚀 Start Invoice Operations Pipeline"
            ):
                with st.spinner(
                    "Processing invoices..."
                ):
                    batch_data = []

                    progress_bar = st.progress(0)

                    session = SessionLocal()

                    for idx, file in enumerate(uploaded_files):
                        try:
                            # SAVE FILE
                            save_uploaded_file(file)

                            # EXTRACT CONTENT
                            payload = extract_content_smart(file)

                            if not payload:
                                continue

                            # AI PROCESSING
                            data = process_invoice_agent(
                                gemini_api_key,
                                payload
                            )

                            # AUDIT ENGINE
                            status, risks, fraud_score = run_compliance_audit(
                                data,
                                st.session_state["fraud_history"],
                                st.session_state["historical_totals"]
                            )

                            st.session_state[
                                "historical_totals"
                            ].append(
                                data["total_amount"]
                            )

                            # DATABASE SAVE
                            record = InvoiceRecord(
                                filename=file.name,
                                vendor_name=data["vendor_name"],
                                invoice_number=data["invoice_number"],
                                invoice_date=data["invoice_date"],
                                currency=data["currency"],
                                subtotal=data["subtotal"],
                                tax=data["vat_or_tax_amount"],
                                total=data["total_amount"],
                                confidence=data["confidence_score"],
                                fraud_score=fraud_score,
                                status=status,
                                risks=", ".join(risks)
                            )

                            session.add(record)

                            session.commit()

                            # UI DATA
                            batch_data.append({
                                "Filename": file.name,
                                "Vendor": data["vendor_name"],
                                "Invoice #": data["invoice_number"],
                                "Currency": data["currency"],
                                "Total": data["total_amount"],
                                "Fraud Score": round(fraud_score, 2),
                                "Confidence": f"{data['confidence_score']*100:.0f}%",
                                "Status": status,
                                "Risks": ", ".join(risks)
                            })

                        except Exception as e:
                            logging.error(str(e))

                            batch_data.append({
                                "Filename": file.name,
                                "Vendor": "ERROR",
                                "Invoice #": "FAILED",
                                "Currency": "N/A",
                                "Total": 0,
                                "Fraud Score": 0,
                                "Confidence": 0,
                                "Status": "FAILED",
                                "Risks": str(e)
                            })

                        progress_bar.progress(
                            (idx + 1) / len(uploaded_files)
                        )

                    # RESULTS
                    if batch_data:
                        st.success(
                            "Invoice processing completed."
                        )

                        df = pd.DataFrame(batch_data)

                        st.subheader(
                            "📂 Enterprise Audit Ledger"
                        )

                        st.dataframe(
                            df,
                            use_container_width=True
                        )

                        # METRICS
                        st.divider()

                        col1, col2, col3 = st.columns(3)

                        flagged_count = len(
                            df[df["Status"] == "⚠️ Flagged"]
                        )

                        avg_confidence = round(
                            df["Confidence"]
                            .str.replace("%", "")
                            .astype(float)
                            .mean(),
                            2
                        )

                        col1.metric(
                            "Processed Invoices",
                            len(df)
                        )

                        col2.metric(
                            "Flagged Invoices",
                            flagged_count
                        )

                        col3.metric(
                            "Average Confidence",
                            f"{avg_confidence}%"
                        )

                        # EXPORTS
                        st.divider()

                        st.subheader(
                            "💾 Export Reports"
                        )

                        ec1, ec2, ec3 = st.columns(3)

                        # CSV
                        csv = df.to_csv(
                            index=False
                        ).encode("utf-8")

                        ec1.download_button(
                            "📄 Download CSV",
                            csv,
                            "invoice_audit.csv",
                            "text/csv",
                            use_container_width=True
                        )

                        # JSON
                        json_data = df.to_json(
                            orient="records"
                        ).encode("utf-8")

                        ec2.download_button(
                            "🤖 Download JSON",
                            json_data,
                            "invoice_audit.json",
                            "application/json",
                            use_container_width=True
                        )

                        # EXCEL
                        output = io.BytesIO()

                        with pd.ExcelWriter(
                            output,
                            engine="openpyxl"
                        ) as writer:
                            df.to_excel(
                                writer,
                                index=False
                            )

                        ec3.download_button(
                            "📊 Download Excel",
                            output.getvalue(),
                            "invoice_audit.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

# =========================================================
# TAB 2
# =========================================================

with tab2:
    st.subheader("📊 Vendor Analytics")

    session = SessionLocal()

    records = session.query(
        InvoiceRecord
    ).all()

    analytics_data = []

    for row in records:
        analytics_data.append({
            "Vendor": row.vendor_name,
            "Total": row.total,
            "Fraud Score": row.fraud_score,
            "Status": row.status
        })

    if analytics_data:
        analytics_df = pd.DataFrame(
            analytics_data
        )

        # BAR CHART
        fig = px.bar(
            analytics_df,
            x="Vendor",
            y="Total",
            color="Status",
            title="Vendor Spend Analytics"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # FRAUD HISTOGRAM
        fig2 = px.histogram(
            analytics_df,
            x="Fraud Score",
            nbins=10,
            title="Fraud Score Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    else:
        st.info(
            "Upload invoices to start generating compliance analytics."
        )

# =========================================================
# TAB 3
# =========================================================

with tab3:
    st.subheader("🗄️ Stored Invoice Database")

    session = SessionLocal()

    rows = session.query(
        InvoiceRecord
    ).all()

    database_records = []

    for row in rows:
        database_records.append({
            "Filename": row.filename,
            "Vendor": row.vendor_name,
            "Invoice": row.invoice_number,
            "Date": row.invoice_date,
            "Currency": row.currency,
            "Total": row.total,
            "Fraud Score": row.fraud_score,
            "Status": row.status,
            "Risks": row.risks
        })

    if database_records:
        db_df = pd.DataFrame(
            database_records
        )

        st.dataframe(
            db_df,
            use_container_width=True
        )

    else:
        st.info(
            "No invoices stored yet."
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built with AI-powered invoice intelligence and compliance automation."
)