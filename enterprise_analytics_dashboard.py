from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


COLUMN_ALIASES = {
    "invoice_id": [
        "invoice_id",
        "invoice_number",
        "invoice_no",
        "invoice",
        "number",
        "id",
    ],
    "vendor": [
        "vendor",
        "vendor_name",
        "supplier",
        "supplier_name",
        "seller",
        "seller_name",
        "merchant",
        "company_name",
    ],
    "currency": [
        "currency",
        "invoice_currency",
        "currency_code",
        "curr",
    ],
    "status": [
        "status",
        "invoice_status",
        "payment_status",
        "approval_status",
        "processing_status",
    ],
    "invoice_date": [
        "invoice_date",
        "date",
        "issue_date",
        "bill_date",
        "billing_date",
        "created_at",
        "uploaded_at",
        "processed_at",
    ],
    "total_amount": [
        "total_amount",
        "invoice_amount",
        "amount",
        "grand_total",
        "total",
        "net_amount",
        "payable_amount",
        "subtotal",
    ],
    "fraud_score": [
        "fraud_score",
        "risk_score",
        "fraud_risk_score",
        "anomaly_score",
        "fraud_probability",
        "risk_percentage",
    ],
    "risk_level": [
        "risk_level",
        "fraud_risk",
        "risk_category",
        "severity",
    ],
    "compliance_status": [
        "compliance_status",
        "gst_status",
        "vat_status",
        "tax_status",
        "compliance_result",
        "validation_status",
        "is_compliant",
        "gst_valid",
        "vat_valid",
    ],
    "high_risk": [
        "high_risk",
        "is_high_risk",
        "flagged",
        "fraud_flag",
        "requires_review",
    ],
}


COMPLIANT_TERMS = ("compliant", "valid", "passed", "pass", "approved", "true", "yes")
NON_COMPLIANT_TERMS = (
    "non-compliant",
    "non compliant",
    "invalid",
    "failed",
    "fail",
    "mismatch",
    "issue",
    "false",
    "no",
)


CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "AED": "AED ",
    "CAD": "C$",
    "AUD": "A$",
    "SGD": "S$",
}


def render_enterprise_analytics_dashboard(
    invoices_df: pd.DataFrame | list[dict[str, Any]],
    gemini_model: Any | None = None,
    *,
    key_prefix: str = "enterprise_analytics",
) -> pd.DataFrame:
    """Render the enterprise analytics dashboard and return the filtered data.

    Pass the same DataFrame you already use for the current analytics tab.
    Optionally pass your configured Gemini model to enrich the insights panel.
    """

    _inject_dashboard_css()
    df = normalize_invoice_dataframe(invoices_df)

    st.markdown(
        """
        <section class="ea-hero">
            <div>
                <p class="ea-eyebrow">Invoice Operations Command Center</p>
                <h1>Executive Analytics Dashboard</h1>
                <p class="ea-subtitle">
                    Spend visibility, compliance posture, and fraud exposure across the invoice lifecycle.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        _render_empty_state()
        return df

    filtered_df = render_analytics_filters(df, key_prefix=key_prefix)

    if filtered_df.empty:
        _render_empty_state(
            title="No invoices match the selected filters",
            body="Adjust the vendor, currency, status, or date filters to restore the analytics view.",
        )
        return filtered_df

    metrics = calculate_kpis(filtered_df)
    render_kpi_cards(metrics)

    render_ai_insights_panel(filtered_df, gemini_model=gemini_model)
    render_dashboard_charts(filtered_df)

    return filtered_df


def normalize_invoice_dataframe(
    invoices_df: pd.DataFrame | list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """Create a dashboard-safe schema without changing the source data."""

    if invoices_df is None:
        return pd.DataFrame()

    df = pd.DataFrame(invoices_df).copy()
    if df.empty:
        return df

    lowered_columns = {str(col).strip().lower(): col for col in df.columns}
    normalized = pd.DataFrame(index=df.index)

    for target_col, aliases in COLUMN_ALIASES.items():
        source_col = _find_column(lowered_columns, aliases)
        if source_col is not None:
            normalized[target_col] = df[source_col]

    normalized["invoice_id"] = _series_or_default(normalized, "invoice_id", df.index)
    normalized["vendor"] = _series_or_default(
        normalized, "vendor", "Unknown Vendor"
    ).fillna("Unknown Vendor")
    normalized["currency"] = _series_or_default(normalized, "currency", "UNKNOWN").fillna(
        "UNKNOWN"
    )
    normalized["country"] = normalized["currency"].map({
        "INR": "India",
        "USD": "United States",
        "GBP": "United Kingdom",
        "EUR": "European Union",
        "AED": "United Arab Emirates",
        "AUD": "Australia",
        "SGD": "Singapore",
        "CAD": "Canada"
    }).fillna("Unknown")

    normalized["tax_system"] = normalized["currency"].map({
        "INR": "GST",
        "USD": "Sales Tax",
        "GBP": "VAT",
        "EUR": "VAT",
        "AED": "VAT",
        "AUD": "GST",
        "SGD": "GST",
        "CAD": "GST/HST"
    }).fillna("Unknown")
    normalized["status"] = _series_or_default(normalized, "status", "Processed").fillna(
        "Processed"
    )
    normalized["invoice_date"] = pd.to_datetime(
        _series_or_default(normalized, "invoice_date", pd.Timestamp.today()), errors="coerce"
    )
    normalized["invoice_date"] = normalized["invoice_date"].fillna(pd.Timestamp.today())
    normalized["total_amount"] = pd.to_numeric(
        _series_or_default(normalized, "total_amount", 0), errors="coerce"
    ).fillna(0)
    normalized["fraud_score"] = pd.to_numeric(
        _series_or_default(normalized, "fraud_score", 0), errors="coerce"
    ).fillna(0)

    if normalized["fraud_score"].max() <= 1:
        normalized["fraud_score"] = normalized["fraud_score"] * 100
    normalized["fraud_score"] = normalized["fraud_score"].clip(lower=0, upper=100)

    normalized["risk_level"] = _series_or_default(normalized, "risk_level", "").fillna("")
    normalized["compliance_status"] = _series_or_default(
        normalized, "compliance_status", "Unknown"
    ).fillna(
        "Unknown"
    )
    normalized["is_compliant"] = normalized["compliance_status"].apply(_parse_compliance_status)
    normalized["high_risk"] = _derive_high_risk_flag(normalized)
    normalized["vendor"] = normalized["vendor"].astype(str).str.strip().replace("", "Unknown Vendor")
    normalized["currency"] = normalized["currency"].astype(str).str.upper().str.strip()
    normalized["currency"] = normalized["currency"].replace("", "UNKNOWN")
    normalized["status"] = normalized["status"].astype(str).str.strip().replace("", "Processed")

    for col in df.columns:
        if col not in normalized.columns:
            normalized[col] = df[col]

    return normalized


def render_analytics_filters(df: pd.DataFrame, *, key_prefix: str) -> pd.DataFrame:
    st.markdown('<div class="ea-section-title">Control Filters</div>', unsafe_allow_html=True)

    min_date = df["invoice_date"].min().date()
    max_date = df["invoice_date"].max().date()

    filter_cols = st.columns([1.35, 1, 1, 1.15])
    with filter_cols[0]:
        vendors = st.multiselect(
            "Vendor",
            options=sorted(df["vendor"].dropna().unique()),
            default=[],
            placeholder="All vendors",
            key=f"{key_prefix}_vendor_filter",
        )
    with filter_cols[1]:
        currencies = st.multiselect(
            "Currency",
            options=sorted(df["currency"].dropna().unique()),
            default=[],
            placeholder="All currencies",
            key=f"{key_prefix}_currency_filter",
        )
    with filter_cols[2]:
        statuses = st.multiselect(
            "Status",
            options=sorted(df["status"].dropna().unique()),
            default=[],
            placeholder="All statuses",
            key=f"{key_prefix}_status_filter",
        )
    with filter_cols[3]:
        selected_dates = st.date_input(
            "Invoice date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_date_filter",
        )

    filtered_df = df.copy()

    if vendors:
        filtered_df = filtered_df[filtered_df["vendor"].isin(vendors)]
    if currencies:
        filtered_df = filtered_df[filtered_df["currency"].isin(currencies)]
    if statuses:
        filtered_df = filtered_df[filtered_df["status"].isin(statuses)]

    start_date, end_date = _coerce_date_range(selected_dates, min_date, max_date)
    filtered_df = filtered_df[
        (filtered_df["invoice_date"].dt.date >= start_date)
        & (filtered_df["invoice_date"].dt.date <= end_date)
    ]

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} invoices from "
        f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}."
    )
    return filtered_df


def calculate_kpis(df: pd.DataFrame) -> dict[str, Any]:
    total_invoices = len(df)
    total_spend = float(df["total_amount"].sum())
    average_invoice_value = float(df["total_amount"].mean()) if total_invoices else 0
    high_risk_count = int(df["high_risk"].sum())
    compliant_known = df["is_compliant"].dropna()
    compliant_count = int(compliant_known.sum()) if not compliant_known.empty else 0
    compliance_percentage = (
        float(compliant_count / len(compliant_known) * 100) if len(compliant_known) else 0
    )
    fraud_risk_percentage = float(high_risk_count / total_invoices * 100) if total_invoices else 0
    vendor_count = int(df["vendor"].nunique())
    currency_context = _currency_context(df)

    return {
        "total_invoices": total_invoices,
        "total_spend": total_spend,
        "average_invoice_value": average_invoice_value,
        "fraud_risk_percentage": fraud_risk_percentage,
        "compliance_percentage": compliance_percentage,
        "vendor_count": vendor_count,
        "high_risk_count": high_risk_count,
        "currency_context": currency_context,
    }


def render_kpi_cards(metrics: dict[str, Any]) -> None:
    st.markdown('<div class="ea-section-title">Executive KPIs</div>', unsafe_allow_html=True)
    spend_value = _format_money(metrics["total_spend"], metrics["currency_context"])
    avg_value = _format_money(metrics["average_invoice_value"], metrics["currency_context"])

    cards = [
        ("Total Invoice Volume", f"{metrics['total_invoices']:,}", "Processed invoices"),
        ("Total Spend", spend_value, "Filtered invoice value"),
        ("Average Invoice Value", avg_value, "Mean invoice amount"),
        ("Fraud Risk Percentage", f"{metrics['fraud_risk_percentage']:.1f}%", "High-risk share"),
        ("Compliance Percentage", f"{metrics['compliance_percentage']:.1f}%", "Validated invoices"),
        ("Vendor Count", f"{metrics['vendor_count']:,}", "Unique suppliers"),
        ("High Risk Invoices", f"{metrics['high_risk_count']:,}", "Require review"),
    ]

    columns = st.columns(4)
    for index, (label, value, helper) in enumerate(cards):
        with columns[index % 4]:
            st.markdown(
                f"""
                <div class="ea-kpi-card">
                    <div class="ea-kpi-label">{label}</div>
                    <div class="ea-kpi-value">{value}</div>
                    <div class="ea-kpi-helper">{helper}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dashboard_charts(df: pd.DataFrame) -> None:
    st.markdown('<div class="ea-section-title">Analytics Workspace</div>', unsafe_allow_html=True)

    trend_df = _build_time_series(df)
    vendor_df = (
        df.groupby("vendor", as_index=False)
        .agg(total_spend=("total_amount", "sum"), invoice_count=("invoice_id", "count"))
        .sort_values("total_spend", ascending=False)
        .head(10)
    )
    compliance_df = _build_compliance_distribution(df)
    currency_df = (
        df.groupby("currency", as_index=False)
        .agg(total_spend=("total_amount", "sum"), invoice_count=("invoice_id", "count"))
        .sort_values("total_spend", ascending=False)
    )
    country_df = (
        df.groupby("country", as_index=False)
        .agg(
            total_spend=("total_amount", "sum"),
            invoice_count=("invoice_id", "count")
        )
        .sort_values("total_spend", ascending=False)
    )
    risk_vendor_df = (
        df.groupby("vendor", as_index=False)
        .agg(
            high_risk_invoices=("high_risk", "sum"),
            average_fraud_score=("fraud_score", "mean"),
            total_spend=("total_amount", "sum"),
        )
        .query("high_risk_invoices > 0 or average_fraud_score >= 50")
        .sort_values(["high_risk_invoices", "average_fraud_score"], ascending=False)
        .head(10)
    )

    left_col, right_col = st.columns([1.45, 1])
    with left_col:
        _render_chart_card(
            "Spend Trend",
            _spend_trend_chart(trend_df),
            "Daily, weekly, or monthly trend based on selected date range.",
        )
    with right_col:
        _render_chart_card(
            "Vendor Leaderboard",
            _vendor_leaderboard_chart(vendor_df),
            "Top vendors by filtered spend.",
        )

    left_col, right_col = st.columns([1, 1])
    with left_col:
        _render_chart_card(
            "Compliance Mix",
            _compliance_pie_chart(compliance_df),
            "Compliant, non-compliant, and unknown validation outcomes.",
        )
    with right_col:
        _render_chart_card(
            "Fraud Score Distribution",
            _fraud_distribution_chart(df),
            "Score spread from low risk to critical review.",
        )

    left_col, right_col = st.columns([1.25, 1])
    with left_col:
        _render_chart_card(
            "Invoice Volume Trend",
            _invoice_volume_chart(trend_df),
            "Operational throughput across the selected period.",
        )
    with right_col:
        _render_chart_card(
            "Currency Distribution",
            _currency_distribution_chart(currency_df),
            "Spend exposure by invoice currency.",
        )
    left_col, right_col = st.columns([1, 1])

    with left_col:
        _render_chart_card(
            "Country Distribution",
            _country_distribution_chart(country_df),
            "Invoice volume distribution by country."
        )

    with right_col:
        _render_chart_card(
            "Country Spend Analysis",
            _country_spend_chart(country_df),
            "Total spend distribution across countries."
        )

    _render_chart_card(
        "High-Risk Vendor Analytics",
        _high_risk_vendor_chart(risk_vendor_df),
        "Vendors ranked by risk concentration and average fraud score.",
    )


def render_ai_insights_panel(df: pd.DataFrame, gemini_model: Any | None = None) -> None:
    insights = generate_business_insights(df)
    gemini_summary = _generate_gemini_summary(df, insights, gemini_model)

    st.markdown('<div class="ea-section-title">AI Insights</div>', unsafe_allow_html=True)
    if gemini_summary:
        st.markdown(
            f"""
            <div class="ea-insight-panel">
                <div class="ea-insight-kicker">Gemini executive summary</div>
                <div class="ea-insight-summary">{gemini_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    insight_cols = st.columns(2)
    for index, insight in enumerate(insights):
        with insight_cols[index % 2]:
            st.markdown(
                f"""
                <div class="ea-insight-card ea-severity-{insight['severity']}">
                    <div class="ea-insight-title">{insight['title']}</div>
                    <div class="ea-insight-body">{insight['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def generate_business_insights(df: pd.DataFrame) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []

    top_vendors = (
        df.groupby("vendor", as_index=False)["total_amount"]
        .sum()
        .sort_values("total_amount", ascending=False)
        .head(3)
    )
    if not top_vendors.empty:
        vendor_text = ", ".join(
            f"{row.vendor} ({_format_compact_number(row.total_amount)})"
            for row in top_vendors.itertuples()
        )
        insights.append(
            {
                "title": "Top vendor concentration",
                "body": f"Highest spend is concentrated with {vendor_text}. Review negotiated rates and recurring purchase patterns.",
                "severity": "info",
            }
        )

    high_risk_rate = float(df["high_risk"].mean() * 100) if len(df) else 0
    if high_risk_rate >= 20:
        severity = "critical"
        body = f"High-risk invoices represent {high_risk_rate:.1f}% of the filtered population. Prioritize manual review before payment release."
    elif high_risk_rate >= 8:
        severity = "warning"
        body = f"Fraud exposure is elevated at {high_risk_rate:.1f}%. Watch repeat vendors and unusual invoice values."
    else:
        severity = "positive"
        body = f"Fraud exposure is controlled at {high_risk_rate:.1f}% high-risk invoices in the selected period."
    insights.append({"title": "Fraud risk posture", "body": body, "severity": severity})

    anomaly = _find_spend_anomaly(df)
    if anomaly:
        insights.append(anomaly)

    non_compliant = df[df["is_compliant"] == False]  # noqa: E712
    compliance_base = df["is_compliant"].dropna()
    if not compliance_base.empty:
        compliance_rate = float(compliance_base.mean() * 100)
        if compliance_rate < 85:
            top_issue_vendors = (
                non_compliant.groupby("vendor", as_index=False)["invoice_id"]
                .count()
                .sort_values("invoice_id", ascending=False)
                .head(3)
            )
            vendor_names = ", ".join(top_issue_vendors["vendor"].astype(str).tolist())
            insights.append(
                {
                    "title": "Compliance issues require attention",
                    "body": f"Compliance is {compliance_rate:.1f}%. Vendors contributing most exceptions: {vendor_names or 'not enough vendor data'}.",
                    "severity": "warning",
                }
            )
        else:
            insights.append(
                {
                    "title": "Healthy compliance coverage",
                    "body": f"Compliance is {compliance_rate:.1f}% across validated invoices, supporting audit readiness.",
                    "severity": "positive",
                }
            )

    fraud_movement = _fraud_movement_insight(df)
    if fraud_movement:
        insights.append(fraud_movement)

    return insights[:6]


def _spend_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        trend_df,
        x="period",
        y="total_spend",
        markers=True,
        labels={"period": "", "total_spend": "Spend"},
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_traces(line_width=3, marker_size=8, hovertemplate="%{x}<br>Spend: %{y:,.2f}<extra></extra>")
    return _style_figure(fig)


def _vendor_leaderboard_chart(vendor_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        vendor_df.sort_values("total_spend"),
        x="total_spend",
        y="vendor",
        orientation="h",
        color="total_spend",
        color_continuous_scale=["#60a5fa", "#22c55e"],
        labels={"total_spend": "Spend", "vendor": ""},
        hover_data={"invoice_count": True, "total_spend": ":,.2f"},
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style_figure(fig, height=390)


def _compliance_pie_chart(compliance_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        compliance_df,
        names="status",
        values="count",
        hole=0.62,
        color="status",
        color_discrete_map={
            "Compliant": "#22c55e",
            "Non-compliant": "#ef4444",
            "Unknown": "#94a3b8",
        },
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.02] * len(compliance_df))
    return _style_figure(fig, height=365)


def _fraud_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="fraud_score",
        nbins=12,
        labels={"fraud_score": "Fraud score"},
        color_discrete_sequence=["#f97316"],
    )
    fig.add_vrect(x0=70, x1=100, fillcolor="#ef4444", opacity=0.13, line_width=0)
    fig.update_traces(hovertemplate="Score band: %{x}<br>Invoices: %{y}<extra></extra>")
    return _style_figure(fig, height=365)


def _invoice_volume_chart(trend_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        trend_df,
        x="period",
        y="invoice_count",
        labels={"period": "", "invoice_count": "Invoices"},
        color_discrete_sequence=["#a78bfa"],
    )
    fig.update_traces(hovertemplate="%{x}<br>Invoices: %{y:,}<extra></extra>")
    return _style_figure(fig)


def _currency_distribution_chart(currency_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        currency_df,
        names="currency",
        values="total_spend",
        hole=0.58,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _style_figure(fig)

def _country_distribution_chart(country_df: pd.DataFrame) -> go.Figure:

    fig = px.pie(
        country_df,
        names="country",
        values="invoice_count",
        hole=0.58
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    return _style_figure(fig)

def _country_spend_chart(country_df: pd.DataFrame) -> go.Figure:

    fig = px.bar(
        country_df,
        x="country",
        y="total_spend",
        text="total_spend"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Total Spend"
    )

    return _style_figure(fig)

def _high_risk_vendor_chart(risk_vendor_df: pd.DataFrame) -> go.Figure:
    if risk_vendor_df.empty:
        risk_vendor_df = pd.DataFrame(
            {"vendor": ["No high-risk vendors"], "high_risk_invoices": [0], "average_fraud_score": [0]}
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=risk_vendor_df["vendor"],
            y=risk_vendor_df["high_risk_invoices"],
            name="High-risk invoices",
            marker_color="#ef4444",
            hovertemplate="%{x}<br>High-risk invoices: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=risk_vendor_df["vendor"],
            y=risk_vendor_df["average_fraud_score"],
            name="Avg fraud score",
            mode="lines+markers",
            yaxis="y2",
            marker_color="#facc15",
            line={"width": 3},
            hovertemplate="%{x}<br>Avg fraud score: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"title": "High-risk invoices"},
        yaxis2={
            "title": "Avg fraud score",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return _style_figure(fig, height=430)


def _build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    span_days = max((df["invoice_date"].max() - df["invoice_date"].min()).days, 1)
    frequency = "D" if span_days <= 45 else "W-MON" if span_days <= 180 else "MS"

    trend_df = (
        df.set_index("invoice_date")
        .resample(frequency)
        .agg(total_spend=("total_amount", "sum"), invoice_count=("invoice_id", "count"))
        .reset_index()
        .rename(columns={"invoice_date": "period"})
    )
    return trend_df


def _build_compliance_distribution(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["is_compliant"].map({True: "Compliant", False: "Non-compliant"}).fillna("Unknown")
    return labels.value_counts().rename_axis("status").reset_index(name="count")


def _render_chart_card(title: str, figure: go.Figure, helper: str) -> None:
    st.markdown(
        f"""
        <div class="ea-chart-heading">
            <span>{title}</span>
            <small>{helper}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font={"color": "#dbeafe", "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 12, "r": 12, "t": 18, "b": 16},
        hoverlabel={"bgcolor": "#0f172a", "font_size": 13, "font_family": "Inter"},
        xaxis={
            "gridcolor": "rgba(148,163,184,0.16)",
            "zerolinecolor": "rgba(148,163,184,0.2)",
        },
        yaxis={
            "gridcolor": "rgba(148,163,184,0.16)",
            "zerolinecolor": "rgba(148,163,184,0.2)",
        },
    )
    return fig


def _generate_gemini_summary(
    df: pd.DataFrame,
    insights: list[dict[str, str]],
    gemini_model: Any | None,
) -> str | None:
    if gemini_model is None:
        return None

    summary_payload = {
        "invoice_count": int(len(df)),
        "total_spend": float(df["total_amount"].sum()),
        "average_invoice_value": float(df["total_amount"].mean()) if len(df) else 0,
        "high_risk_percentage": float(df["high_risk"].mean() * 100) if len(df) else 0,
        "compliance_percentage": float(df["is_compliant"].dropna().mean() * 100)
        if not df["is_compliant"].dropna().empty
        else None,
        "top_vendors": (
            df.groupby("vendor")["total_amount"].sum().sort_values(ascending=False).head(5).to_dict()
        ),
        "rule_based_insights": insights,
    }
    prompt = (
        "You are an enterprise invoice operations analyst. Write a concise executive "
        "summary in 3 bullet-style sentences using the aggregate dashboard data below. "
        "Mention risk, spend concentration, and compliance actions. Do not invent values.\n\n"
        f"{json.dumps(summary_payload, default=str)}"
    )

    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", None)
        return text.strip() if text else None
    except Exception:
        return None


def _find_spend_anomaly(df: pd.DataFrame) -> dict[str, str] | None:
    if len(df) < 5:
        return None

    amount_mean = df["total_amount"].mean()
    amount_std = df["total_amount"].std()
    if pd.isna(amount_std) or amount_std == 0:
        return None

    threshold = amount_mean + (2 * amount_std)
    anomalous = df[df["total_amount"] > threshold].sort_values("total_amount", ascending=False)
    if anomalous.empty:
        return {
            "title": "Spend anomaly scan",
            "body": "No invoice values are more than two standard deviations above the current filtered average.",
            "severity": "positive",
        }

    top = anomalous.iloc[0]
    return {
        "title": "Spend anomaly detected",
        "body": f"{top['vendor']} has an invoice at {_format_compact_number(top['total_amount'])}, materially above the filtered average.",
        "severity": "warning",
    }


def _fraud_movement_insight(df: pd.DataFrame) -> dict[str, str] | None:
    if df["invoice_date"].nunique() < 4:
        return None

    ordered = df.sort_values("invoice_date")
    midpoint = ordered["invoice_date"].min() + (
        ordered["invoice_date"].max() - ordered["invoice_date"].min()
    ) / 2
    previous = ordered[ordered["invoice_date"] < midpoint]
    current = ordered[ordered["invoice_date"] >= midpoint]
    if previous.empty or current.empty:
        return None

    previous_rate = previous["high_risk"].mean() * 100
    current_rate = current["high_risk"].mean() * 100
    delta = current_rate - previous_rate

    if delta >= 5:
        return {
            "title": "Fraud risk is increasing",
            "body": f"High-risk share rose from {previous_rate:.1f}% to {current_rate:.1f}% across the selected period.",
            "severity": "critical",
        }
    if delta <= -5:
        return {
            "title": "Fraud risk is improving",
            "body": f"High-risk share improved from {previous_rate:.1f}% to {current_rate:.1f}% across the selected period.",
            "severity": "positive",
        }
    return None


def _find_column(lowered_columns: dict[str, Any], aliases: list[str]) -> Any | None:
    for alias in aliases:
        if alias in lowered_columns:
            return lowered_columns[alias]
    return None


def _series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _parse_compliance_status(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if any(term in normalized for term in NON_COMPLIANT_TERMS):
        return False
    if any(term in normalized for term in COMPLIANT_TERMS):
        return True
    return None


def _derive_high_risk_flag(df: pd.DataFrame) -> pd.Series:
    explicit = df.get("high_risk")
    if explicit is not None:
        explicit_flag = explicit.astype(str).str.lower().isin(["true", "1", "yes", "y", "high"])
    else:
        explicit_flag = pd.Series(False, index=df.index)

    risk_level_flag = df["risk_level"].astype(str).str.lower().str.contains("high|critical|severe")
    score_flag = df["fraud_score"] >= 70
    return explicit_flag | risk_level_flag | score_flag


def _coerce_date_range(selected_dates: Any, min_date: date, max_date: date) -> tuple[date, date]:
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        return selected_dates
    if isinstance(selected_dates, list) and len(selected_dates) == 2:
        return selected_dates[0], selected_dates[1]
    if isinstance(selected_dates, date):
        return selected_dates, selected_dates
    return min_date, max_date


def _currency_context(df: pd.DataFrame) -> str:
    currencies = sorted(df["currency"].dropna().unique())
    if len(currencies) == 1:
        return currencies[0]
    return "MIXED"


def _format_money(value: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, "" if currency == "MIXED" else f"{currency} ")
    suffix = "" if currency != "MIXED" else " mixed"
    return f"{symbol}{value:,.0f}{suffix}"


def _format_compact_number(value: float) -> str:
    value = float(value)
    if abs(value) >= 10_000_000:
        return f"{value / 10_000_000:.2f}Cr"
    if abs(value) >= 100_000:
        return f"{value / 100_000:.2f}L"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def _render_empty_state(
    *,
    title: str = "No invoice analytics available yet",
    body: str = "Upload and process invoices to populate the executive dashboard.",
) -> None:
    st.markdown(
        f"""
        <div class="ea-empty-state">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 10% 0%, rgba(56, 189, 248, 0.14), transparent 30%),
                    linear-gradient(135deg, #020617 0%, #0f172a 44%, #111827 100%);
                color: #e5eefb;
            }

            .block-container {
                max-width: 1480px;
                padding-top: 1.8rem;
                padding-bottom: 3rem;
            }

            .ea-hero {
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.72));
                border-radius: 8px;
                padding: 1.35rem 1.5rem;
                margin-bottom: 1.15rem;
                box-shadow: 0 24px 70px rgba(2, 6, 23, 0.28);
            }

            .ea-eyebrow {
                color: #67e8f9;
                font-size: 0.78rem;
                letter-spacing: 0;
                text-transform: uppercase;
                margin: 0 0 0.25rem 0;
                font-weight: 700;
            }

            .ea-hero h1 {
                color: #f8fafc;
                font-size: clamp(1.75rem, 2.6vw, 2.75rem);
                line-height: 1.05;
                margin: 0;
                letter-spacing: 0;
            }

            .ea-subtitle {
                color: #b6c7db;
                max-width: 860px;
                margin: 0.65rem 0 0 0;
                font-size: 1rem;
            }

            .ea-section-title {
                color: #f8fafc;
                font-weight: 750;
                font-size: 1.02rem;
                margin: 1rem 0 0.65rem 0;
            }

            .ea-kpi-card {
                min-height: 132px;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.62));
                border-radius: 8px;
                padding: 1rem 1rem 0.9rem 1rem;
                margin-bottom: 0.85rem;
                box-shadow: 0 18px 38px rgba(2, 6, 23, 0.22);
            }

            .ea-kpi-label {
                color: #9fb3c8;
                font-size: 0.78rem;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: 0;
            }

            .ea-kpi-value {
                color: #ffffff;
                font-size: clamp(1.35rem, 2vw, 2.1rem);
                font-weight: 800;
                margin-top: 0.38rem;
                line-height: 1.1;
                overflow-wrap: anywhere;
            }

            .ea-kpi-helper {
                color: #93a4b8;
                font-size: 0.85rem;
                margin-top: 0.45rem;
            }

            .ea-chart-heading {
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-bottom: 0;
                background: rgba(15, 23, 42, 0.68);
                border-radius: 8px 8px 0 0;
                padding: 0.8rem 0.95rem 0.3rem 0.95rem;
                margin-top: 0.35rem;
            }

            .ea-chart-heading span {
                display: block;
                color: #f8fafc;
                font-weight: 720;
                font-size: 0.98rem;
            }

            .ea-chart-heading small {
                display: block;
                color: #9fb3c8;
                font-size: 0.8rem;
                margin-top: 0.15rem;
            }

            .ea-insight-panel,
            .ea-insight-card,
            .ea-empty-state {
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: rgba(15, 23, 42, 0.78);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 0.85rem;
            }

            .ea-insight-kicker {
                color: #67e8f9;
                font-size: 0.78rem;
                font-weight: 750;
                text-transform: uppercase;
                letter-spacing: 0;
                margin-bottom: 0.4rem;
            }

            .ea-insight-summary {
                color: #dbeafe;
                line-height: 1.55;
            }

            .ea-insight-title {
                color: #f8fafc;
                font-weight: 760;
                margin-bottom: 0.35rem;
            }

            .ea-insight-body {
                color: #b6c7db;
                font-size: 0.92rem;
                line-height: 1.48;
            }

            .ea-severity-critical {
                border-left: 4px solid #ef4444;
            }

            .ea-severity-warning {
                border-left: 4px solid #f59e0b;
            }

            .ea-severity-positive {
                border-left: 4px solid #22c55e;
            }

            .ea-severity-info {
                border-left: 4px solid #38bdf8;
            }

            .ea-empty-state h3 {
                color: #f8fafc;
                margin: 0 0 0.3rem 0;
            }

            .ea-empty-state p {
                color: #b6c7db;
                margin: 0;
            }

            div[data-testid="stDateInput"] input,
            div[data-testid="stMultiSelect"] {
                color: #e5eefb;
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .ea-hero {
                    padding: 1rem;
                }

                .ea-kpi-card {
                    min-height: 116px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
