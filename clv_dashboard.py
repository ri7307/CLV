"""
CLV Business Intelligence Dashboard
====================================
Multi-page interactive Dash dashboard built from the CLV Analytics notebook.

Run:  python clv_dashboard.py
Then open:  http://localhost:8050

Requirements:
    pip install dash dash-bootstrap-components plotly pandas numpy

Dataset:
    Place "CLV Dataset.csv" in the same directory as this file,
    or set the path in DATA_PATH below.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "CLV Dataset.csv"   # ← change path if needed

# Theme colours (matches the notebook palette)
BG       = "#0f172a"
SURFACE  = "#1e293b"
SURFACE2 = "#0f2038"
BLUE     = "#1a73e8"
GREEN    = "#34a853"
YELLOW   = "#fbbc04"
RED      = "#ea4335"
PURPLE   = "#9334e6"
CYAN     = "#06b6d4"
FONT     = "#e2e8f0"
MUTED    = "#94a3b8"
ACCENT   = "#93c5fd"
PALETTE  = [BLUE, GREEN, YELLOW, RED, PURPLE, CYAN]
GRID     = "#1e3a5f"

LAYOUT_BASE = dict(
    plot_bgcolor=BG,
    paper_bgcolor=SURFACE,
    font=dict(color=FONT, family="DM Sans, Inter, sans-serif"),
    title_font=dict(size=16, color=ACCENT, family="DM Sans, Inter, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT)),
    margin=dict(l=55, r=30, t=75, b=55),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)

# ── Data Loading & Preprocessing ─────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        # Generate synthetic data so the dashboard works without the CSV
        print(f"[INFO] '{DATA_PATH}' not found — generating synthetic demo data.")
        rng = np.random.default_rng(42)
        n = 15_000
        regions   = ["North", "South", "East", "West", "Central"]
        channels  = ["Referral", "Organic", "Paid Search", "Social Media", "Email"]
        incomes   = ["High", "Medium", "Low"]
        genders   = ["Male", "Female", "Other"]
        products  = ["Electronics", "Fashion", "Home & Living", "Sports", "Books",
                     "Beauty", "Grocery", "Toys"]

        df = pd.DataFrame({
            "purchase_frequency":         rng.integers(1, 20, n),
            "average_order_value":        rng.uniform(10, 800, n),
            "total_revenue_generated":    rng.uniform(50, 1500, n),
            "recency_last_purchase_days": rng.uniform(1, 365, n),
            "customer_tenure_days":       rng.uniform(30, 2000, n),
            "website_visit_frequency":    rng.integers(1, 50, n),
            "session_duration_minutes":   rng.uniform(1, 120, n),
            "pages_per_session":          rng.integers(1, 30, n),
            "cart_wishlist_activity":     rng.uniform(0, 10, n),
            "email_open_rate":            rng.uniform(0, 1, n),
            "age":                        rng.uniform(18, 70, n),
            "gender":                     rng.choice(genders, n),
            "location_region":            rng.choice(regions, n),
            "income_level":               rng.choice(incomes, n),
            "acquisition_channel":        rng.choice(channels, n),
            "campaign_response_rate":     rng.uniform(0, 1, n),
            "coupon_usage_frequency":     rng.integers(0, 20, n),
            "loyalty_program_membership": rng.integers(0, 2, n),
            "referral_activity":          rng.integers(0, 10, n),
            "product_category_preference":rng.choice(products, n),
            "unique_products_purchased":  rng.uniform(1, 15, n),
            "repeat_purchase_rate":       rng.uniform(0, 1, n),
            "discount_sensitivity":       rng.uniform(0, 1, n),
            "subscription_purchase_behavior": rng.integers(0, 2, n),
            "customer_support_tickets":   rng.integers(0, 10, n),
            "complaint_frequency":        rng.integers(0, 5, n),
            "customer_satisfaction_score":rng.uniform(1, 10, n),
            "return_refund_rate":         rng.uniform(0, 0.5, n),
            "churn_probability":          rng.uniform(0, 1, n),
            "engagement_score":           rng.uniform(0, 100, n),
            "Customer_Lifetime_Value":    rng.uniform(300, 8000, n),
        })
        return df

    df = pd.read_csv(DATA_PATH)
    # Basic cleaning
    critical = ["Customer_Lifetime_Value", "churn_probability",
                "total_revenue_generated", "purchase_frequency",
                "average_order_value", "engagement_score"]
    df = df.dropna(subset=critical).drop_duplicates()
    df = df[
        (df["Customer_Lifetime_Value"] > 0) &
        (df["churn_probability"].between(0, 1)) &
        (df["purchase_frequency"] >= 1) &
        (df["average_order_value"] > 0)
    ].copy()

    # Notebook-aligned targeted imputation for non-critical analytical fields.
    median_cols = [
        "recency_last_purchase_days", "cart_wishlist_activity", "age",
        "discount_sensitivity", "unique_products_purchased",
    ]
    for col in median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = [
        "gender", "location_region", "income_level",
        "acquisition_channel", "product_category_preference",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df


def engineer_features(df):
    df = df.copy()

    # CLV Segment: primary notebook pipeline uses 33rd/66th percentile thresholds.
    lbls = ["Low Value", "Medium Value", "High Value"]
    try:
        df["clv_segment"] = pd.qcut(
            df["Customer_Lifetime_Value"], q=[0, 0.33, 0.66, 1],
            labels=lbls, duplicates="drop",
        ).astype(str)
    except ValueError:
        bins = [0, 2000, 5000, df["Customer_Lifetime_Value"].max() + 1]
        df["clv_segment"] = pd.cut(
            df["Customer_Lifetime_Value"], bins=bins,
            labels=lbls, include_lowest=True,
        ).astype(str)

    # Keep the fixed notebook chart thresholds available as a separate lens.
    fixed_bins = [0, 2000, 5000, df["Customer_Lifetime_Value"].max() + 1]
    df["clv_fixed_segment"] = pd.cut(
        df["Customer_Lifetime_Value"], bins=fixed_bins,
        labels=lbls, include_lowest=True,
    ).astype(str)

    # Churn Risk
    df["churn_risk"] = pd.cut(df["churn_probability"],
                              bins=[0, 0.35, 0.65, 1.01],
                              labels=["Low Risk", "Medium Risk", "High Risk"],
                              include_lowest=True).astype(str)

    # Revenue Band
    rev_bins  = [0, 100, 500, 1000, df["total_revenue_generated"].max() + 1]
    rev_lbls  = ["Bronze", "Silver", "Gold", "Platinum"]
    df["revenue_band"] = pd.cut(df["total_revenue_generated"],
                                bins=rev_bins, labels=rev_lbls,
                                include_lowest=True).astype(str)

    # Normalised engagement
    mn, mx = df["engagement_score"].min(), df["engagement_score"].max()
    if mx == mn:
        df["engagement_score_normalized"] = 50
    else:
        df["engagement_score_normalized"] = (
            (df["engagement_score"] - mn) / (mx - mn) * 100
        ).round(2)

    # Notebook formula: service priority balances CLV, engagement, and retained probability.
    df["priority_score"] = (
        df["Customer_Lifetime_Value"] *
        df["engagement_score_normalized"] *
        (1 - df["churn_probability"]) / 1000
    ).round(3)

    # Retention urgency is intentionally separate: it highlights value currently at risk.
    df["retention_urgency_score"] = (
        df["Customer_Lifetime_Value"] *
        df["churn_probability"] *
        (1 + (100 - df["engagement_score_normalized"]) / 100)
    ).round(2)

    df["expected_clv_at_risk"] = (
        df["Customer_Lifetime_Value"] * df["churn_probability"]
    ).round(2)
    df["est_refund_revenue_at_risk"] = (
        df["total_revenue_generated"] * df["return_refund_rate"]
    ).round(2)

    df["discount_tier"] = pd.cut(
        df["discount_sensitivity"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low Sensitivity", "Medium Sensitivity", "High Sensitivity"],
        include_lowest=True,
    ).astype(str)

    df["recency_tier"] = pd.cut(
        df["recency_last_purchase_days"],
        bins=[-0.01, 30, 90, np.inf],
        labels=["Active (<=30 days)", "Drifting (31-90 days)", "Dormant (>90 days)"],
        include_lowest=True,
    ).astype(str)

    df["product_breadth"] = pd.cut(
        df["unique_products_purchased"],
        bins=[-0.01, 3, 7, np.inf],
        labels=["Low Breadth (1-3)", "Medium Breadth (4-7)", "High Breadth (8+)"],
        include_lowest=True,
    ).astype(str)

    df["loyalty_label"] = df["loyalty_program_membership"].map(
        {0: "Non-Member", 1: "Member"}
    ).fillna("Unknown")
    df["subscription_label"] = df["subscription_purchase_behavior"].map(
        {0: "No Subscription", 1: "Subscription Buyer"}
    ).fillna("Unknown")

    return df


raw_df = load_data()
df     = engineer_features(raw_df)

# Unique filter values
ALL_CHANNELS = sorted(df["acquisition_channel"].dropna().unique().tolist())
ALL_REGIONS  = sorted(df["location_region"].dropna().unique().tolist())
ALL_PRODUCTS = sorted(df["product_category_preference"].dropna().unique().tolist())
SEGMENT_ORDER = ["High Value", "Medium Value", "Low Value"]
RISK_ORDER = ["High Risk", "Medium Risk", "Low Risk"]
REVENUE_ORDER = ["Platinum", "Gold", "Silver", "Bronze"]
RECENCY_ORDER = ["Active (<=30 days)", "Drifting (31-90 days)", "Dormant (>90 days)"]
DISCOUNT_ORDER = ["Low Sensitivity", "Medium Sensitivity", "High Sensitivity"]
BREADTH_ORDER = ["Low Breadth (1-3)", "Medium Breadth (4-7)", "High Breadth (8+)"]

# ── App Setup ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="CLV Intelligence Dashboard",
)
server = app.server  # expose Flask server for deployment (Gunicorn / Render / etc.)

# ── Reusable style helpers ─────────────────────────────────────────────────────
def card(children, style=None, className=""):
    base = {
        "background": SURFACE,
        "borderRadius": "12px",
        "padding": "20px",
        "border": f"1px solid #1e3a5f",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base, className=className)


def insight_box(text):
    """Small coloured insight strip under each chart."""
    return html.Div(
        html.P(text, style={"margin": 0, "fontSize": "0.82rem", "lineHeight": "1.5",
                            "color": "#cbd5e1"}),
        style={
            "background": "#0f2038",
            "borderLeft": f"3px solid {BLUE}",
            "borderRadius": "0 6px 6px 0",
            "padding": "9px 14px",
            "marginTop": "8px",
        },
    )


def kpi_card(label, value, color, icon=""):
    return html.Div(
        [
            html.Div(f"{icon} {value}", style={
                "color": color, "fontSize": "1.7rem", "fontWeight": "700",
                "letterSpacing": "-0.02em",
            }),
            html.Div(label, style={"color": MUTED, "fontSize": "0.75rem",
                                   "marginTop": "4px", "fontWeight": "500"}),
        ],
        style={
            "background": SURFACE2,
            "border": f"1px solid {color}33",
            "borderRadius": "10px",
            "padding": "16px 20px",
            "minWidth": "150px",
            "flex": "1",
            "textAlign": "center",
        },
    )


def section_title(text):
    return html.H5(text, style={"color": ACCENT, "fontWeight": "600",
                                "marginBottom": "4px", "letterSpacing": "-0.01em"})


def empty_figure(title="No data for selected filters"):
    fig = go.Figure()
    fig.add_annotation(
        text=title, x=0.5, y=0.5, showarrow=False,
        font=dict(color=MUTED, size=16),
    )
    fig.update_layout(**LAYOUT_BASE)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def filter_dataset(source, clv="All", churn="All", channel="All", region="All",
                   product="All", income="All"):
    d = source.copy()
    if clv != "All":
        d = d[d["clv_segment"] == clv]
    if churn != "All":
        d = d[d["churn_risk"] == churn]
    if channel != "All":
        d = d[d["acquisition_channel"] == channel]
    if region != "All":
        d = d[d["location_region"] == region]
    if product != "All":
        d = d[d["product_category_preference"] == product]
    if income != "All":
        d = d[d["income_level"] == income]
    return d


def metric_note(title, value, detail, color=BLUE):
    return html.Div(
        [
            html.Div(title, style={"color": MUTED, "fontSize": "0.72rem",
                                   "fontWeight": "600", "textTransform": "uppercase"}),
            html.Div(value, style={"color": color, "fontSize": "1.25rem",
                                   "fontWeight": "700", "marginTop": "4px"}),
            html.Div(detail, style={"color": "#cbd5e1", "fontSize": "0.78rem",
                                    "lineHeight": "1.35", "marginTop": "5px"}),
        ],
        style={
            "background": SURFACE2,
            "border": f"1px solid {color}33",
            "borderRadius": "8px",
            "padding": "14px 16px",
            "flex": "1",
            "minWidth": "210px",
        },
    )


# ── Sidebar & Navigation ──────────────────────────────────────────────────────
NAV_ITEMS = [
    ("🏠", "Executive Overview",       "/"),
    ("📡", "Channel & Revenue Intel",  "/channel"),
    ("🔴", "Churn & Retention",        "/churn"),
    ("🛒", "Product & Category",       "/product"),
    ("🌍", "Regional & Demographics",  "/regional"),
    ("⭐", "Engagement & Loyalty",     "/engagement"),
    ("☁️", "Big Data Risk Lab",        "/risk-lab"),
    ("📋", "Data Explorer",            "/explorer"),
]


def sidebar():
    return html.Div(
        [
            html.Div(
                [
                    html.Div("📊", style={"fontSize": "2rem"}),
                    html.Div(
                        [
                            html.Div("CLV Intel", style={
                                "fontWeight": "700", "fontSize": "1.1rem",
                                "color": "#ffffff", "lineHeight": "1.2",
                            }),
                            html.Div("Dashboard", style={
                                "color": ACCENT, "fontSize": "0.72rem",
                                "letterSpacing": "0.12em", "textTransform": "uppercase",
                            }),
                        ]
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px",
                       "padding": "24px 20px 20px"},
            ),
            html.Hr(style={"borderColor": "#1e3a5f", "margin": "0 16px 12px"}),
            html.Div(
                [
                    dcc.Link(
                        html.Div(
                            [
                                html.Span(icon, style={"fontSize": "1.1rem", "width": "22px"}),
                                html.Span(label, style={"fontSize": "0.85rem", "fontWeight": "500"}),
                            ],
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "10px", "padding": "10px 16px",
                                   "borderRadius": "8px", "cursor": "pointer"},
                            id=f"nav-{path.strip('/') or 'home'}",
                            className="nav-item",
                        ),
                        href=path,
                        style={"textDecoration": "none", "color": FONT},
                    )
                    for icon, label, path in NAV_ITEMS
                ],
                style={"padding": "0 8px"},
            ),
            html.Div(
                [
                    html.Hr(style={"borderColor": "#1e3a5f"}),
                    html.Div(
                        [
                            html.Div(f"Total Customers: {len(df):,}",
                                     style={"color": MUTED, "fontSize": "0.72rem"}),
                            html.Div("Dataset: CLV Analytics",
                                     style={"color": MUTED, "fontSize": "0.72rem"}),
                        ],
                        style={"padding": "0 16px 16px"},
                    ),
                ],
                style={"marginTop": "auto"},
            ),
        ],
        style={
            "width": "220px",
            "minWidth": "220px",
            "background": SURFACE,
            "height": "100vh",
            "position": "sticky",
            "top": "0",
            "display": "flex",
            "flexDirection": "column",
            "borderRight": "1px solid #1e3a5f",
            "overflowY": "auto",
        },
    )


# ── Main Layout ───────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                sidebar(),
                html.Div(
                    id="page-content",
                    style={
                        "flex": "1",
                        "overflowY": "auto",
                        "padding": "28px 32px",
                        "background": BG,
                        "minHeight": "100vh",
                    },
                ),
            ],
            style={"display": "flex", "fontFamily": "DM Sans, Inter, sans-serif"},
        ),
    ],
    style={"background": BG, "color": FONT},
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; background: ''' + BG + '''; color: ''' + FONT + '''; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ''' + BG + '''; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        .nav-item:hover { background: #1e3a5f !important; }
        .Select-control { background: ''' + SURFACE + ''' !important; border-color: #334155 !important; color: ''' + FONT + ''' !important; }
        .Select-menu-outer { background: ''' + SURFACE + ''' !important; }
        .Select-option { color: ''' + FONT + ''' !important; }
        .Select-option.is-focused { background: #1e3a5f !important; }
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
            background: ''' + SURFACE + ''' !important;
            color: ''' + FONT + ''' !important;
            border-color: #1e3a5f !important;
        }
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
            background: #0f2038 !important;
            color: ''' + ACCENT + ''' !important;
            font-weight: 600;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''


# ── Shared filter widget ───────────────────────────────────────────────────────
def filter_row(id_prefix, show_channel=True, show_churn=True, show_region=False):
    items = []
    if show_churn:
        items.append(
            html.Div([
                html.Label("CLV Segment", style={"color": MUTED, "fontSize": "0.75rem",
                                                  "marginBottom": "4px"}),
                dcc.Dropdown(
                    id=f"{id_prefix}-clv",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High Value", "Medium Value", "Low Value"]],
                    value="All",
                    clearable=False,
                    style={"background": SURFACE, "color": FONT},
                ),
            ], style={"flex": "1"}),
        )
        items.append(
            html.Div([
                html.Label("Churn Risk", style={"color": MUTED, "fontSize": "0.75rem",
                                                "marginBottom": "4px"}),
                dcc.Dropdown(
                    id=f"{id_prefix}-churn",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High Risk", "Medium Risk", "Low Risk"]],
                    value="All",
                    clearable=False,
                    style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
        )
    if show_channel:
        items.append(
            html.Div([
                html.Label("Acquisition Channel", style={"color": MUTED, "fontSize": "0.75rem",
                                                          "marginBottom": "4px"}),
                dcc.Dropdown(
                    id=f"{id_prefix}-channel",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": x, "value": x} for x in ALL_CHANNELS],
                    value="All",
                    clearable=False,
                    style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
        )
    if show_region:
        items.append(
            html.Div([
                html.Label("Region", style={"color": MUTED, "fontSize": "0.75rem",
                                            "marginBottom": "4px"}),
                dcc.Dropdown(
                    id=f"{id_prefix}-region",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": x, "value": x} for x in ALL_REGIONS],
                    value="All",
                    clearable=False,
                    style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
        )
    return html.Div(items, style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                                  "marginBottom": "20px"})


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────
def page_overview():
    return html.Div([
        section_title("🏠 Executive Overview"),
        html.P("Filter the entire customer base and view top-level KPIs, segment mix, and risk summary.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        filter_row("ov"),

        # KPI row
        html.Div(id="ov-kpis", style={"display": "flex", "gap": "12px",
                                       "flexWrap": "wrap", "marginBottom": "24px"}),

        # Charts row 1
        html.Div([
            html.Div([
                card([
                    html.Div("CLV Segment Distribution", style={"color": ACCENT,
                             "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="ov-pie", config={"displayModeBar": False}),
                    insight_box("Most customers cluster in the Medium Value band. "
                                "Focus upsell campaigns on this group to migrate them "
                                "into High Value — each conversion lifts projected revenue significantly."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                card([
                    html.Div("Churn Risk Breakdown", style={"color": ACCENT,
                             "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="ov-churn-bar", config={"displayModeBar": False}),
                    insight_box("Over 26% of customers are at High Churn Risk. "
                                "Immediate outreach — personalised offers or satisfaction surveys — "
                                "can recover a significant portion before they defect."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        # Charts row 2
        html.Div([
            html.Div([
                card([
                    html.Div("Revenue Band Composition", style={"color": ACCENT,
                             "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="ov-revenue-donut", config={"displayModeBar": False}),
                    insight_box("Platinum customers (₹1 000+) are a tiny but critical cohort. "
                                "Dedicated account management and exclusive perks for this group "
                                "protects disproportionate revenue."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                card([
                    html.Div("CLV vs Churn Probability (Sample)", style={"color": ACCENT,
                             "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="ov-scatter", config={"displayModeBar": False}),
                    insight_box("The upper-right quadrant (high CLV + high churn) is the "
                                "'Danger Zone'. These customers demand urgent, prioritised "
                                "retention intervention before revenue impact materialises."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — Channel & Revenue Intel
# ─────────────────────────────────────────────────────────────────────────────
def page_channel():
    return html.Div([
        section_title("📡 Channel & Revenue Intel"),
        html.P("Identify which acquisition channels produce the highest CLV and revenue contribution.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        filter_row("ch", show_churn=True, show_channel=False),

        html.Div([
            html.Div([
                card([
                    html.Div("Avg CLV by Acquisition Channel", style={"color": ACCENT,
                             "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="ch-bar", config={"displayModeBar": False}),
                    insight_box("Referral is consistently the best-performing channel by CLV. "
                                "Investing in referral programs (incentivised invites, "
                                "referral bonuses) yields customers who spend more over their lifetime."),
                ])
            ], style={"flex": "1", "minWidth": "320px"}),
            html.Div([
                card([
                    html.Div("Revenue Band Distribution by Channel", style={"color": ACCENT,
                             "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="ch-stacked", config={"displayModeBar": False}),
                    insight_box("Channels with a higher share of Platinum/Gold customers "
                                "should receive proportionally larger marketing budgets — "
                                "quality of acquisition matters more than volume."),
                ])
            ], style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div([
            card([
                html.Div("Purchase Frequency vs Average Order Value by Channel",
                         style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                dcc.Graph(id="ch-bubble", config={"displayModeBar": False}),
                insight_box("Bubble size = customer count. Channels in the top-right corner "
                            "(high frequency + high AOV) are the growth engines; prioritise "
                            "retention spend here to compound lifetime value."),
            ])
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — Churn & Retention
# ─────────────────────────────────────────────────────────────────────────────
def page_churn():
    return html.Div([
        section_title("🔴 Churn & Retention Planner"),
        html.P("Identify at-risk high-value customers and plan proactive retention interventions.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        html.Div([
            html.Div([
                html.Label("Top N Danger Zone Customers", style={"color": MUTED,
                           "fontSize": "0.75rem", "marginBottom": "4px"}),
                dcc.Slider(id="churn-n", min=5, max=50, step=5, value=20,
                           marks={i: str(i) for i in range(5, 55, 10)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"flex": "2", "minWidth": "280px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "24px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Churn Probability Distribution by CLV Segment",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="churn-violin", config={"displayModeBar": False}),
                    insight_box("High Value customers show a wide spread of churn probability — "
                                "meaning many are still recoverable. A personalised retention "
                                "campaign targeting those above 0.5 churn probability in this tier "
                                "offers the highest ROI."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                card([
                    html.Div("CLV Segment × Churn Risk Heatmap (Customer Count)",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="churn-heat", config={"displayModeBar": False}),
                    insight_box("The High Value + High Risk cell is the single most urgent "
                                "business problem. Even retaining 10% of this cohort prevents "
                                "a measurable revenue decline."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        card([
            html.Div("⚠️ Danger Zone — High Value × High Churn Risk Customers",
                     style={"color": RED, "fontWeight": "700", "marginBottom": "12px",
                            "fontSize": "1rem"}),
            html.Div(id="churn-table"),
            insight_box("Sort by Retention Urgency Score to determine intervention order. "
                        "High satisfaction scores in this list indicate the customer can still "
                        "be saved — a targeted offer within 48 hours significantly reduces churn."),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — Product & Category
# ─────────────────────────────────────────────────────────────────────────────
def page_product():
    return html.Div([
        section_title("🛒 Product & Category Insights"),
        html.P("Explore which product preferences and breadth correlate with the highest CLV.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        html.Div([
            html.Div([
                html.Label("Income Level", style={"color": MUTED, "fontSize": "0.75rem",
                                                   "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="prod-income",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High", "Medium", "Low"]],
                    value="All", clearable=False,
                    style={"background": SURFACE},
                ),
            ], style={"flex": "1", "maxWidth": "220px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div(id="prod-summary", style={"display": "flex", "gap": "12px",
                                            "flexWrap": "wrap", "marginBottom": "20px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Avg CLV by Product Category",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="prod-clv-bar", config={"displayModeBar": False},
                              style={"height": "430px"}),
                    insight_box("Categories with the highest CLV should receive premium "
                                "shelf placement, targeted loyalty rewards, and early-access "
                                "promotions to deepen the customer's category affinity."),
                ])
            ], style={"flex": "1", "minWidth": "420px"}),
            html.Div([
                card([
                    html.Div("Avg Revenue by Product Category",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="prod-rev-bar", config={"displayModeBar": False},
                              style={"height": "430px"}),
                    insight_box("Gap between CLV rank and Revenue rank in the same category "
                                "reveals undermonetised relationships — strong signals for "
                                "bundle offers and upsell campaigns."),
                ])
            ], style={"flex": "1", "minWidth": "420px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        card([
            html.Div("Discount Sensitivity vs Repeat Purchase Rate by Category",
                     style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
            dcc.Graph(id="prod-scatter", config={"displayModeBar": False}),
            insight_box("Categories in the low-discount + high-repeat zone are your most "
                        "loyal segments — reward them with exclusive access, not discounts. "
                        "High-discount + low-repeat categories need value perception work."),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5 — Regional & Demographics
# ─────────────────────────────────────────────────────────────────────────────
def page_regional():
    return html.Div([
        section_title("🌍 Regional & Demographic Analysis"),
        html.P("Compare CLV, churn, and engagement across geographies and demographic segments.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Avg CLV & Churn Probability by Region (Dual Axis)",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="reg-dual", config={"displayModeBar": False}),
                    insight_box("Regions where CLV bars are tall but the churn line is also high "
                                "are your highest-risk, highest-value markets — deploy localised "
                                "retention campaigns here before revenue declines."),
                ])
            ], style={"flex": "1", "minWidth": "320px"}),
            html.Div([
                card([
                    html.Div("CLV Distribution by Income Level",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="reg-box", config={"displayModeBar": False}),
                    insight_box("High-income customers have a wider CLV range, meaning there is "
                                "greater upside to convert them from Medium to High Value — "
                                "premium product bundles and white-glove service are effective levers."),
                ])
            ], style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Gender Distribution Across CLV Segments",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="reg-gender", config={"displayModeBar": False}),
                    insight_box("Significant gender skew within High Value segments suggests "
                                "targeted communication styles and product emphasis can improve "
                                "conversion efficiency and reduce unnecessary ad spend."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                card([
                    html.Div("Average Satisfaction Score by Region & Churn Risk",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="reg-sat", config={"displayModeBar": False}),
                    insight_box("Low satisfaction in High Risk regions is a compounding risk "
                                "factor. Proactive NPS surveys and faster support SLAs "
                                "in these regions can arrest churn before it accelerates."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 6 — Engagement & Loyalty
# ─────────────────────────────────────────────────────────────────────────────
def page_engagement():
    return html.Div([
        section_title("⭐ Engagement & Loyalty Analysis"),
        html.P("Understand how engagement, loyalty membership, and support interaction "
               "shape CLV and retention.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Engagement Score Distribution by CLV Segment",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="eng-hist", config={"displayModeBar": False}),
                    insight_box("High Value customers consistently show higher normalised "
                                "engagement. Investing in engagement touchpoints — personalised "
                                "email, push notifications, loyalty milestones — directly lifts CLV."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                card([
                    html.Div("Email Open Rate vs CLV by Loyalty Membership",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="eng-scatter", config={"displayModeBar": False}),
                    insight_box("Loyalty members with higher email open rates have "
                                "meaningfully higher CLV — email nurture sequences and "
                                "loyalty tier communications are high-ROI retention tools."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Avg CLV by Loyalty Program Membership",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="eng-loyalty", config={"displayModeBar": False}),
                    insight_box("Loyalty members generate significantly higher CLV. "
                                "Reducing the barrier to join the loyalty program "
                                "and actively migrating non-members is a high-impact retention strategy."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                card([
                    html.Div("Support Tickets vs CLV (Avg by Segment)",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="eng-support", config={"displayModeBar": False}),
                    insight_box("Frequent support interactions for High Value customers signal "
                                "unresolved friction — proactive outreach before a ticket is "
                                "raised prevents satisfaction decay and protects CLV."),
                ])
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 7 — Big Data Risk Lab
# ─────────────────────────────────────────────────────────────────────────────
def page_risk_lab():
    return html.Div([
        section_title("☁️ Big Data Risk Lab"),
        html.P("Notebook-backed operational lab for RFM recency, discount sensitivity, "
               "product breadth, return/refund risk, and Spark-style pipeline monitoring.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        html.Div([
            html.Div([
                html.Label("CLV Segment", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="risk-clv",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High Value", "Medium Value", "Low Value"]],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Churn Risk", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="risk-churn",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High Risk", "Medium Risk", "Low Risk"]],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Channel", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="risk-channel",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": x, "value": x} for x in ALL_CHANNELS],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Region", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="risk-region",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": x, "value": x} for x in ALL_REGIONS],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Business Lens", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="risk-lens",
                    options=[
                        {"label": "Recency Tier", "value": "recency_tier"},
                        {"label": "Discount Sensitivity", "value": "discount_tier"},
                        {"label": "Product Breadth", "value": "product_breadth"},
                        {"label": "Revenue Band", "value": "revenue_band"},
                        {"label": "Acquisition Channel", "value": "acquisition_channel"},
                    ],
                    value="recency_tier", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div(id="risk-kpis", style={"display": "flex", "gap": "12px",
                                         "flexWrap": "wrap", "marginBottom": "20px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("RFM Recency × CLV Segment Heatmap",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="risk-rfm-heat", config={"displayModeBar": False}),
                    insight_box("Dormant High Value customers are the notebook's clearest "
                                "win-back opportunity: they have proven value, but recency shows "
                                "relationship decay before churn fully materialises."),
                ])
            ], style={"flex": "1", "minWidth": "340px"}),
            html.Div([
                card([
                    html.Div("Return/Refund Revenue-at-Risk Matrix",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="risk-return-heat", config={"displayModeBar": False}),
                    insight_box("High return/refund exposure inside High Value or High Risk cells "
                                "points to product-fit and service issues. Fixing those issues "
                                "protects revenue and reduces avoidable support cost."),
                ])
            ], style={"flex": "1", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Discount Sensitivity: CLV vs Return Rate",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="risk-discount", config={"displayModeBar": False}),
                    insight_box("If high-discount customers show lower CLV or higher return rates, "
                                "promotion spend is attracting margin-diluting buyers. Shift offers "
                                "toward loyalty perks and bundles for low-sensitivity customers."),
                ])
            ], style={"flex": "1", "minWidth": "340px"}),
            html.Div([
                card([
                    html.Div("Product Breadth Loyalty Profile",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="risk-breadth", config={"displayModeBar": False}),
                    insight_box("Customers buying across more categories are stickier and more "
                                "valuable. Medium breadth customers are ideal cross-sell targets "
                                "because they already show ecosystem adoption."),
                ])
            ], style={"flex": "1", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div([
            html.Div([
                card([
                    html.Div("Selected Business Lens: Volume, CLV, Churn",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="risk-lens-chart", config={"displayModeBar": False}),
                    insight_box("Use the lens selector to switch the same filtered data between "
                                "recency, discount, breadth, revenue, and acquisition views. This "
                                "turns the notebook aggregations into an interactive decision tool."),
                ])
            ], style={"flex": "1", "minWidth": "340px"}),
            html.Div([
                card([
                    html.Div("Spark / Cloud Analytics Pipeline",
                             style={"color": ACCENT, "fontWeight": "600", "marginBottom": "4px"}),
                    dcc.Graph(id="risk-pipeline", config={"displayModeBar": False}),
                    insight_box("This mirrors the notebook flow: ingest CSV, clean and impute, "
                                "engineer CLV/risk features, aggregate at business dimensions, "
                                "then serve the interactive dashboard for decision teams."),
                ])
            ], style={"flex": "1", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 8 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer():
    return html.Div([
        section_title("📋 Data Explorer"),
        html.P("Browse and filter the processed customer dataset directly.",
               style={"color": MUTED, "marginBottom": "20px", "fontSize": "0.88rem"}),

        html.Div([
            html.Div([
                html.Label("CLV Segment", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="exp-clv",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High Value", "Medium Value", "Low Value"]],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Churn Risk", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="exp-churn",
                    options=[{"label": x, "value": x}
                             for x in ["All", "High Risk", "Medium Risk", "Low Risk"]],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Revenue Band", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="exp-band",
                    options=[{"label": x, "value": x}
                             for x in ["All", "Platinum", "Gold", "Silver", "Bronze"]],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Region", style={"color": MUTED, "fontSize": "0.75rem"}),
                dcc.Dropdown(
                    id="exp-region",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": x, "value": x} for x in ALL_REGIONS],
                    value="All", clearable=False, style={"background": SURFACE},
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.Div(id="exp-count", style={"color": MUTED, "fontSize": "0.82rem",
                                         "marginBottom": "10px"}),
        html.Div(id="exp-table"),
    ])


# ── Page Router ───────────────────────────────────────────────────────────────
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    routes = {
        "/":           page_overview,
        "/channel":    page_channel,
        "/churn":      page_churn,
        "/product":    page_product,
        "/regional":   page_regional,
        "/engagement": page_engagement,
        "/risk-lab":    page_risk_lab,
        "/explorer":   page_explorer,
    }
    fn = routes.get(path, page_overview)
    return fn()


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Executive Overview
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("ov-kpis", "children"),
    Output("ov-pie", "figure"),
    Output("ov-churn-bar", "figure"),
    Output("ov-revenue-donut", "figure"),
    Output("ov-scatter", "figure"),
    Input("ov-clv", "value"),
    Input("ov-churn", "value"),
    Input("ov-channel", "value"),
)
def cb_overview(clv_seg, churn_seg, channel):
    d = filter_dataset(df, clv=clv_seg, churn=churn_seg, channel=channel)
    if d.empty:
        empty = empty_figure()
        return [], empty, empty, empty, empty

    # KPI cards
    total     = len(d)
    avg_clv   = d["Customer_Lifetime_Value"].mean()
    danger    = ((d["clv_segment"] == "High Value") & (d["churn_risk"] == "High Risk")).sum()
    avg_churn = d["churn_probability"].mean()
    best_ch   = d.groupby("acquisition_channel")["Customer_Lifetime_Value"].mean().idxmax()
    kpis = [
        kpi_card("Total Customers",        f"{total:,}",        BLUE,   "👥"),
        kpi_card("Avg CLV",                f"₹{avg_clv:,.0f}",  GREEN,  "💰"),
        kpi_card("Danger Zone Customers",  f"{danger:,}",       RED,    "🔴"),
        kpi_card("Avg Churn Probability",  f"{avg_churn:.1%}",  YELLOW, "📉"),
        kpi_card("Best Acquisition Channel", best_ch,           PURPLE, "🏆"),
    ]

    # CLV Donut
    seg_cnt = d["clv_segment"].value_counts()
    colour_map = {"High Value": BLUE, "Medium Value": GREEN, "Low Value": RED}
    colours = [colour_map.get(l, CYAN) for l in seg_cnt.index]
    pie = go.Figure(go.Pie(
        labels=seg_cnt.index, values=seg_cnt.values, hole=0.45,
        marker=dict(colors=colours, line=dict(color=BG, width=3)),
        textinfo="label+percent",
    ))
    pie.update_layout(title="CLV Segment", **LAYOUT_BASE)

    # Churn bar
    cr_cnt = d["churn_risk"].value_counts().reindex(
        ["High Risk", "Medium Risk", "Low Risk"]).dropna()
    churn_bar = go.Figure(go.Bar(
        x=cr_cnt.index, y=cr_cnt.values,
        marker_color=[RED, YELLOW, GREEN],
        text=cr_cnt.values, texttemplate="%{text:,}", textposition="outside",
    ))
    churn_bar.update_layout(title="Churn Risk Count", **LAYOUT_BASE)

    # Revenue band donut
    rb_cnt = d["revenue_band"].value_counts()
    band_colours = {"Platinum": PURPLE, "Gold": YELLOW, "Silver": BLUE, "Bronze": MUTED}
    rev_donut = go.Figure(go.Pie(
        labels=rb_cnt.index, values=rb_cnt.values, hole=0.45,
        marker=dict(colors=[band_colours.get(b, CYAN) for b in rb_cnt.index],
                    line=dict(color=BG, width=3)),
        textinfo="label+percent",
    ))
    rev_donut.update_layout(title="Revenue Band", **LAYOUT_BASE)

    # Scatter (sample)
    sample = d.sample(min(2000, len(d)), random_state=42)
    seg_colour = {"High Value": BLUE, "Medium Value": GREEN, "Low Value": RED}
    scatter = px.scatter(
        sample, x="churn_probability", y="Customer_Lifetime_Value",
        color="clv_segment",
        color_discrete_map=seg_colour,
        size="engagement_score_normalized",
        opacity=0.55, size_max=11,
        labels={"churn_probability": "Churn Probability",
                "Customer_Lifetime_Value": "CLV (₹)"},
    )
    scatter.add_vrect(x0=0.65, x1=1.0, fillcolor=RED, opacity=0.08,
                      line_width=0, annotation_text="Danger Zone",
                      annotation_position="top left",
                      annotation_font_color=RED)
    scatter.update_layout(title="CLV vs Churn Probability", **LAYOUT_BASE)

    return kpis, pie, churn_bar, rev_donut, scatter


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Channel & Revenue
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("ch-bar", "figure"),
    Output("ch-stacked", "figure"),
    Output("ch-bubble", "figure"),
    Input("ch-clv", "value"),
    Input("ch-churn", "value"),
)
def cb_channel(clv_seg, churn_seg):
    d = df.copy()
    if clv_seg   != "All": d = d[d["clv_segment"] == clv_seg]
    if churn_seg != "All": d = d[d["churn_risk"]   == churn_seg]

    ch_agg = d.groupby("acquisition_channel")["Customer_Lifetime_Value"].mean().sort_values()
    bar = go.Figure(go.Bar(
        x=ch_agg.values, y=ch_agg.index, orientation="h",
        marker_color=BLUE,
        text=ch_agg.values.round(0), texttemplate="₹%{text:,.0f}", textposition="outside",
    ))
    bar.update_layout(title="Avg CLV by Channel", **LAYOUT_BASE)

    # Stacked bar: revenue band per channel
    stk = d.groupby(["acquisition_channel", "revenue_band"]).size().unstack(fill_value=0)
    band_order = [b for b in ["Platinum", "Gold", "Silver", "Bronze"] if b in stk.columns]
    stk = stk[band_order]
    band_col = {"Platinum": PURPLE, "Gold": YELLOW, "Silver": BLUE, "Bronze": MUTED}
    stacked = go.Figure()
    for band in band_order:
        stacked.add_trace(go.Bar(
            x=stk.index, y=stk[band], name=band,
            marker_color=band_col[band],
        ))
    stacked.update_layout(barmode="stack", title="Revenue Band by Channel", **LAYOUT_BASE)

    # Bubble: freq vs AOV by channel
    bub = d.groupby("acquisition_channel").agg(
        freq=("purchase_frequency", "mean"),
        aov=("average_order_value", "mean"),
        count=("Customer_Lifetime_Value", "count"),
    ).reset_index()
    bubble = px.scatter(
        bub, x="freq", y="aov", size="count", color="acquisition_channel",
        color_discrete_sequence=PALETTE,
        text="acquisition_channel",
        labels={"freq": "Avg Purchase Frequency", "aov": "Avg Order Value (₹)"},
        size_max=55,
    )
    bubble.update_traces(textposition="top center")
    bubble.update_layout(title="Purchase Frequency vs AOV by Channel", **LAYOUT_BASE)

    return bar, stacked, bubble


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Churn & Retention
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("churn-violin", "figure"),
    Output("churn-heat", "figure"),
    Output("churn-table", "children"),
    Input("churn-n", "value"),
)
def cb_churn(n):
    # Violin
    violin = px.violin(
        df, x="clv_segment", y="churn_probability",
        color="clv_segment",
        color_discrete_map={"High Value": BLUE, "Medium Value": GREEN, "Low Value": RED},
        box=True, points=False,
        labels={"churn_probability": "Churn Probability", "clv_segment": "CLV Segment"},
    )
    violin.update_layout(title="Churn Probability by CLV Segment", **LAYOUT_BASE,
                         showlegend=False)

    # Heatmap
    pivot = df.pivot_table(index="clv_segment", columns="churn_risk",
                           values="Customer_Lifetime_Value", aggfunc="count").fillna(0)
    col_order = [c for c in ["High Risk", "Medium Risk", "Low Risk"] if c in pivot.columns]
    row_order  = [r for r in ["High Value", "Medium Value", "Low Value"] if r in pivot.index]
    pivot = pivot.reindex(index=row_order, columns=col_order, fill_value=0)
    heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values.astype(int), texttemplate="%{text:,}",
        showscale=True,
    ))
    heat.update_layout(title="Customer Count: CLV Segment × Churn Risk", **LAYOUT_BASE)

    # Danger zone table
    danger = df[(df["clv_segment"] == "High Value") & (df["churn_risk"] == "High Risk")].copy()
    cols   = ["Customer_Lifetime_Value", "churn_probability", "engagement_score_normalized",
              "customer_satisfaction_score", "acquisition_channel",
              "location_region", "retention_urgency_score", "priority_score"]
    cols   = [c for c in cols if c in danger.columns]
    danger = danger[cols].sort_values("retention_urgency_score", ascending=False).head(n).round(3)
    danger.columns = [c.replace("_", " ").title() for c in danger.columns]

    table = dash_table.DataTable(
        data=danger.to_dict("records"),
        columns=[{"name": c, "id": c} for c in danger.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": SURFACE2, "color": ACCENT, "fontWeight": "700",
                      "border": "1px solid #1e3a5f"},
        style_cell={"backgroundColor": SURFACE, "color": FONT, "border": "1px solid #1e3a5f",
                    "fontSize": "0.8rem", "textAlign": "left", "padding": "8px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#0f2038"},
        ],
    )
    return violin, heat, table


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Product & Category
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("prod-summary", "children"),
    Output("prod-clv-bar", "figure"),
    Output("prod-rev-bar", "figure"),
    Output("prod-scatter", "figure"),
    Input("prod-income", "value"),
)
def cb_product(income):
    d = df.copy()
    if income != "All":
        d = d[d["income_level"] == income]

    cat_agg = d.groupby("product_category_preference").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_rev=("total_revenue_generated", "mean"),
        avg_disc=("discount_sensitivity", "mean"),
        avg_repeat=("repeat_purchase_rate", "mean"),
        avg_return=("return_refund_rate", "mean"),
        avg_churn=("churn_probability", "mean"),
        customers=("Customer_Lifetime_Value", "count"),
    ).reset_index()

    if cat_agg.empty:
        empty = empty_figure()
        return [], empty, empty, empty

    clv_top = cat_agg.loc[cat_agg["avg_clv"].idxmax()]
    rev_top = cat_agg.loc[cat_agg["avg_rev"].idxmax()]
    loyal_top = cat_agg.loc[cat_agg["avg_repeat"].idxmax()]
    low_return = cat_agg.loc[cat_agg["avg_return"].idxmin()]
    summary = [
        metric_note("Top CLV Category", clv_top["product_category_preference"],
                    f"Avg CLV ₹{clv_top['avg_clv']:,.0f} across {clv_top['customers']:,.0f} customers.",
                    BLUE),
        metric_note("Revenue Leader", rev_top["product_category_preference"],
                    f"Avg revenue ₹{rev_top['avg_rev']:,.0f}; useful for near-term sales focus.",
                    GREEN),
        metric_note("Highest Repeat Rate", loyal_top["product_category_preference"],
                    f"Repeat purchase rate {loyal_top['avg_repeat']:.1%}; strong loyalty signal.",
                    YELLOW),
        metric_note("Lowest Return Rate", low_return["product_category_preference"],
                    f"Return/refund rate {low_return['avg_return']:.1%}; lower operational leakage.",
                    PURPLE),
    ]

    cat_clv = cat_agg.sort_values("avg_clv", ascending=True).copy()
    cat_clv["label"] = cat_clv["avg_clv"].map(lambda v: f"₹{v:,.0f}")

    clv_bar = go.Figure(go.Bar(
        x=cat_clv["avg_clv"], y=cat_clv["product_category_preference"],
        orientation="h",
        marker=dict(color=cat_clv["avg_clv"], colorscale="Blues", showscale=False,
                    line=dict(color=ACCENT, width=1)),
        text=cat_clv["label"],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Avg CLV: ₹%{x:,.0f}<extra></extra>",
    ))
    clv_bar.update_layout(title="Avg CLV by Product Category", **LAYOUT_BASE)
    clv_bar.update_layout(
        height=430,
        margin=dict(l=155, r=110, t=70, b=55),
        xaxis=dict(gridcolor=GRID, title="Average CLV (₹)",
                   range=[0, cat_clv["avg_clv"].max() * 1.22]),
        yaxis=dict(gridcolor=GRID, title="", automargin=True),
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )

    cat_rev = cat_agg.sort_values("avg_rev", ascending=True).copy()
    cat_rev["label"] = cat_rev["avg_rev"].map(lambda v: f"₹{v:,.0f}")
    rev_bar = go.Figure(go.Bar(
        x=cat_rev["avg_rev"], y=cat_rev["product_category_preference"],
        orientation="h",
        marker=dict(color=cat_rev["avg_rev"], colorscale="Greens", showscale=False,
                    line=dict(color="#86efac", width=1)),
        text=cat_rev["label"],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Avg Revenue: ₹%{x:,.0f}<extra></extra>",
    ))
    rev_bar.update_layout(title="Avg Revenue by Product Category", **LAYOUT_BASE)
    rev_bar.update_layout(
        height=430,
        margin=dict(l=155, r=110, t=70, b=55),
        xaxis=dict(gridcolor=GRID, title="Average Revenue (₹)",
                   range=[0, cat_rev["avg_rev"].max() * 1.24]),
        yaxis=dict(gridcolor=GRID, title="", automargin=True),
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )

    scatter = px.scatter(
        cat_agg, x="avg_disc", y="avg_repeat",
        size="avg_clv", color="product_category_preference",
        color_discrete_sequence=PALETTE,
        text="product_category_preference",
        labels={"avg_disc": "Avg Discount Sensitivity",
                "avg_repeat": "Avg Repeat Purchase Rate"},
        size_max=50,
    )
    scatter.update_traces(textposition="top center")
    scatter.update_layout(title="Discount Sensitivity vs Repeat Rate by Category",
                          **LAYOUT_BASE, showlegend=False)

    return summary, clv_bar, rev_bar, scatter


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Regional & Demographics
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("reg-dual", "figure"),
    Output("reg-box", "figure"),
    Output("reg-gender", "figure"),
    Output("reg-sat", "figure"),
    Input("url", "pathname"),
)
def cb_regional(path):
    if path != "/regional":
        raise dash.exceptions.PreventUpdate

    reg = df.groupby("location_region").agg(
        avg_CLV=("Customer_Lifetime_Value", "mean"),
        avg_churn=("churn_probability", "mean"),
    ).reset_index()

    dual = make_subplots(specs=[[{"secondary_y": True}]])
    dual.add_trace(go.Bar(x=reg["location_region"], y=reg["avg_CLV"],
                          name="Avg CLV (₹)", marker_color=BLUE), secondary_y=False)
    dual.add_trace(go.Scatter(x=reg["location_region"], y=reg["avg_churn"],
                              name="Avg Churn Prob", mode="lines+markers",
                              marker=dict(size=10, color=YELLOW),
                              line=dict(color=YELLOW, width=2)), secondary_y=True)
    dual.update_layout(title="CLV & Churn by Region", **LAYOUT_BASE)
    dual.update_yaxes(title_text="Avg CLV (₹)", secondary_y=False, gridcolor=GRID)
    dual.update_yaxes(title_text="Avg Churn Probability", secondary_y=True)

    box = px.box(df, x="income_level", y="Customer_Lifetime_Value",
                 color="income_level",
                 color_discrete_map={"High": BLUE, "Medium": GREEN, "Low": RED},
                 points="outliers",
                 labels={"Customer_Lifetime_Value": "CLV (₹)"})
    box.update_layout(title="CLV Distribution by Income Level", **LAYOUT_BASE,
                      showlegend=False)

    if "gender" in df.columns:
        gen_grp = df.groupby(["clv_segment", "gender"]).size().reset_index(name="count")
        gender_fig = px.bar(gen_grp, x="clv_segment", y="count", color="gender",
                            barmode="group", color_discrete_sequence=PALETTE,
                            labels={"count": "Customer Count", "clv_segment": "CLV Segment"})
        gender_fig.update_layout(title="Gender by CLV Segment", **LAYOUT_BASE)
    else:
        gender_fig = go.Figure().update_layout(**LAYOUT_BASE,
                                               title="Gender data not available")

    sat_agg = df.groupby(["location_region", "churn_risk"])[
        "customer_satisfaction_score"].mean().reset_index()
    sat_fig = px.bar(sat_agg, x="location_region", y="customer_satisfaction_score",
                     color="churn_risk",
                     color_discrete_map={"High Risk": RED,
                                         "Medium Risk": YELLOW, "Low Risk": GREEN},
                     barmode="group",
                     labels={"customer_satisfaction_score": "Avg Satisfaction Score"})
    sat_fig.update_layout(title="Satisfaction by Region & Churn Risk", **LAYOUT_BASE)

    return dual, box, gender_fig, sat_fig


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Engagement & Loyalty
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("eng-hist", "figure"),
    Output("eng-scatter", "figure"),
    Output("eng-loyalty", "figure"),
    Output("eng-support", "figure"),
    Input("url", "pathname"),
)
def cb_engagement(path):
    if path != "/engagement":
        raise dash.exceptions.PreventUpdate

    hist = px.histogram(
        df, x="engagement_score_normalized", color="clv_segment",
        color_discrete_map={"High Value": BLUE, "Medium Value": GREEN, "Low Value": RED},
        nbins=40, barmode="overlay", opacity=0.72,
        labels={"engagement_score_normalized": "Normalised Engagement Score (0–100)"},
    )
    hist.update_layout(title="Engagement Score Distribution by CLV Segment",
                       **LAYOUT_BASE)

    sample = df.sample(min(3000, len(df)), random_state=0)
    sample["loyalty_label"] = sample["loyalty_program_membership"].map(
        {0: "Non-Member", 1: "Member"})
    scatter = px.scatter(
        sample, x="email_open_rate", y="Customer_Lifetime_Value",
        color="loyalty_label",
        color_discrete_map={"Member": BLUE, "Non-Member": MUTED},
        opacity=0.5, size_max=6,
        labels={"email_open_rate": "Email Open Rate",
                "Customer_Lifetime_Value": "CLV (₹)"},
    )
    scatter.update_layout(title="Email Open Rate vs CLV by Loyalty Membership",
                          **LAYOUT_BASE)

    loyalty_agg = df.groupby("loyalty_program_membership")[
        "Customer_Lifetime_Value"].mean().reset_index()
    loyalty_agg["label"] = loyalty_agg["loyalty_program_membership"].map(
        {0: "Non-Member", 1: "Member"})
    loyalty_fig = go.Figure(go.Bar(
        x=loyalty_agg["label"], y=loyalty_agg["Customer_Lifetime_Value"],
        marker_color=[MUTED, BLUE],
        text=loyalty_agg["Customer_Lifetime_Value"].round(0),
        texttemplate="₹%{text:,.0f}", textposition="outside",
    ))
    loyalty_fig.update_layout(title="Avg CLV: Loyalty Members vs Non-Members",
                              **LAYOUT_BASE)

    sup_agg = df.groupby("clv_segment").agg(
        avg_tickets=("customer_support_tickets", "mean"),
        avg_clv=("Customer_Lifetime_Value", "mean"),
    ).reset_index()
    support_fig = px.bar(
        sup_agg, x="clv_segment", y="avg_tickets",
        color="clv_segment",
        color_discrete_map={"High Value": BLUE, "Medium Value": GREEN, "Low Value": RED},
        text=sup_agg["avg_tickets"].round(2),
        labels={"avg_tickets": "Avg Support Tickets", "clv_segment": "CLV Segment"},
    )
    support_fig.update_layout(title="Avg Support Tickets by CLV Segment",
                              **LAYOUT_BASE, showlegend=False)

    return hist, scatter, loyalty_fig, support_fig


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Big Data Risk Lab
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("risk-kpis", "children"),
    Output("risk-rfm-heat", "figure"),
    Output("risk-return-heat", "figure"),
    Output("risk-discount", "figure"),
    Output("risk-breadth", "figure"),
    Output("risk-lens-chart", "figure"),
    Output("risk-pipeline", "figure"),
    Input("risk-clv", "value"),
    Input("risk-churn", "value"),
    Input("risk-channel", "value"),
    Input("risk-region", "value"),
    Input("risk-lens", "value"),
)
def cb_risk_lab(clv_seg, churn_seg, channel, region, lens):
    d = filter_dataset(df, clv=clv_seg, churn=churn_seg,
                       channel=channel, region=region)
    if d.empty:
        empty = empty_figure()
        return [], empty, empty, empty, empty, empty, empty

    high_risk = d[d["churn_risk"] == "High Risk"]
    danger = d[(d["clv_segment"] == "High Value") & (d["churn_risk"] == "High Risk")]
    dormant_hv = d[(d["clv_segment"] == "High Value") &
                   (d["recency_tier"] == "Dormant (>90 days)")]
    refund_risk = d["est_refund_revenue_at_risk"].sum()
    at_risk_clv = high_risk["expected_clv_at_risk"].sum()

    kpis = [
        metric_note("Filtered Cohort", f"{len(d):,}",
                    "Rows currently powering every chart on this page.", BLUE),
        metric_note("Expected CLV at Risk", f"₹{at_risk_clv:,.0f}",
                    "Sum of CLV x churn probability for high-risk customers.", RED),
        metric_note("Danger Zone Value", f"₹{danger['Customer_Lifetime_Value'].sum():,.0f}",
                    "Total CLV in the High Value x High Risk retention queue.", YELLOW),
        metric_note("Dormant High Value", f"{len(dormant_hv):,}",
                    "Win-back candidates with strong lifetime value and weak recency.", PURPLE),
        metric_note("Refund Exposure", f"₹{refund_risk:,.0f}",
                    "Estimated operational leakage from revenue x return/refund rate.", CYAN),
    ]

    rfm = d.pivot_table(
        index="recency_tier", columns="clv_segment",
        values="Customer_Lifetime_Value", aggfunc="count",
    ).reindex(index=RECENCY_ORDER, columns=SEGMENT_ORDER, fill_value=0).fillna(0)
    rfm_heat = go.Figure(go.Heatmap(
        z=rfm.values, x=rfm.columns.tolist(), y=rfm.index.tolist(),
        colorscale="Blues", text=rfm.values.astype(int),
        texttemplate="%{text:,}",
        hovertemplate="Recency: %{y}<br>CLV Segment: %{x}<br>Customers: %{z:,}<extra></extra>",
    ))
    rfm_heat.update_layout(title="Customer Count by Recency and CLV Segment", **LAYOUT_BASE)

    refund = d.pivot_table(
        index="clv_segment", columns="churn_risk",
        values="est_refund_revenue_at_risk", aggfunc="mean",
    ).reindex(index=SEGMENT_ORDER, columns=RISK_ORDER, fill_value=0).fillna(0)
    return_heat = go.Figure(go.Heatmap(
        z=refund.values, x=refund.columns.tolist(), y=refund.index.tolist(),
        colorscale="Reds", text=np.round(refund.values, 1),
        texttemplate="₹%{text:,.1f}",
        hovertemplate="Segment: %{y}<br>Risk: %{x}<br>Avg Refund Exposure: ₹%{z:,.2f}<extra></extra>",
    ))
    return_heat.update_layout(title="Avg Refund Exposure by CLV and Churn Risk", **LAYOUT_BASE)

    disc = d.groupby("discount_tier").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_return=("return_refund_rate", "mean"),
        customers=("Customer_Lifetime_Value", "count"),
    ).reindex(DISCOUNT_ORDER).dropna(how="all").reset_index()
    discount_fig = make_subplots(specs=[[{"secondary_y": True}]])
    discount_fig.add_trace(go.Bar(
        x=disc["discount_tier"], y=disc["avg_clv"],
        name="Avg CLV", marker_color=BLUE,
        text=disc["avg_clv"].round(0), texttemplate="₹%{text:,.0f}",
    ), secondary_y=False)
    discount_fig.add_trace(go.Scatter(
        x=disc["discount_tier"], y=disc["avg_return"],
        name="Avg Return Rate", mode="lines+markers",
        marker=dict(size=10, color=RED), line=dict(color=RED, width=2),
    ), secondary_y=True)
    discount_fig.update_layout(title="Discount Sensitivity Economics", **LAYOUT_BASE)
    discount_fig.update_yaxes(title_text="Avg CLV (₹)", gridcolor=GRID, secondary_y=False)
    discount_fig.update_yaxes(title_text="Avg Return Rate", secondary_y=True)

    breadth = d.groupby("product_breadth").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_frequency=("purchase_frequency", "mean"),
        avg_churn=("churn_probability", "mean"),
        customers=("Customer_Lifetime_Value", "count"),
    ).reindex(BREADTH_ORDER).dropna(how="all").reset_index()
    breadth_fig = px.scatter(
        breadth, x="avg_frequency", y="avg_clv", size="customers",
        color="avg_churn", text="product_breadth",
        color_continuous_scale=["#34a853", "#fbbc04", "#ea4335"],
        labels={"avg_frequency": "Avg Purchase Frequency",
                "avg_clv": "Avg CLV (₹)", "avg_churn": "Avg Churn Probability"},
        size_max=55,
    )
    breadth_fig.update_traces(textposition="top center")
    breadth_fig.update_layout(title="Product Breadth as Loyalty Proxy", **LAYOUT_BASE)

    lens_labels = {
        "recency_tier": RECENCY_ORDER,
        "discount_tier": DISCOUNT_ORDER,
        "product_breadth": BREADTH_ORDER,
        "revenue_band": REVENUE_ORDER,
    }
    lens_agg = d.groupby(lens).agg(
        customers=("Customer_Lifetime_Value", "count"),
        avg_churn=("churn_probability", "mean"),
    )
    if lens in lens_labels:
        lens_agg = lens_agg.reindex(lens_labels[lens]).dropna(how="all")
    else:
        lens_agg = lens_agg.join(
            d.groupby(lens)["Customer_Lifetime_Value"].mean().rename("avg_clv")
        ).sort_values("avg_clv", ascending=False).drop(columns=["avg_clv"])
    lens_agg = lens_agg.reset_index()

    lens_fig = make_subplots(specs=[[{"secondary_y": True}]])
    lens_fig.add_trace(go.Bar(
        x=lens_agg[lens], y=lens_agg["customers"],
        name="Customers", marker_color=CYAN,
        text=lens_agg["customers"], texttemplate="%{text:,}",
    ), secondary_y=False)
    lens_fig.add_trace(go.Scatter(
        x=lens_agg[lens], y=lens_agg["avg_churn"],
        name="Avg Churn", mode="lines+markers",
        marker=dict(size=9, color=YELLOW), line=dict(color=YELLOW, width=2),
    ), secondary_y=True)
    lens_fig.update_layout(title=f"{lens.replace('_', ' ').title()} Lens", **LAYOUT_BASE)
    lens_fig.update_yaxes(title_text="Customers", gridcolor=GRID, secondary_y=False)
    lens_fig.update_yaxes(title_text="Avg Churn Probability", secondary_y=True)

    pipeline = go.Figure(go.Funnel(
        y=[
            "CSV Ingest",
            "Clean + Impute",
            "Feature Engineering",
            "Business Aggregations",
            "Interactive Dashboard",
        ],
        x=[len(raw_df), len(df), len(df), len(d), len(d)],
        marker=dict(color=[BLUE, GREEN, YELLOW, PURPLE, CYAN]),
        textinfo="value+percent initial",
        hovertemplate="%{y}<br>Rows: %{x:,}<extra></extra>",
    ))
    pipeline.update_layout(title="Notebook-to-Dashboard Pipeline Throughput", **LAYOUT_BASE)

    return kpis, rfm_heat, return_heat, discount_fig, breadth_fig, lens_fig, pipeline


# ═════════════════════════════════════════════════════════════════════════════
#  CALLBACKS — Data Explorer
# ═════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("exp-count", "children"),
    Output("exp-table", "children"),
    Input("exp-clv", "value"),
    Input("exp-churn", "value"),
    Input("exp-band", "value"),
    Input("exp-region", "value"),
)
def cb_explorer(clv_seg, churn_seg, band, region):
    d = df.copy()
    if clv_seg   != "All": d = d[d["clv_segment"]    == clv_seg]
    if churn_seg != "All": d = d[d["churn_risk"]      == churn_seg]
    if band      != "All": d = d[d["revenue_band"]    == band]
    if region    != "All": d = d[d["location_region"] == region]

    display_cols = [
        "clv_segment", "churn_risk", "revenue_band",
        "Customer_Lifetime_Value", "churn_probability",
        "total_revenue_generated", "purchase_frequency",
        "average_order_value", "engagement_score_normalized",
        "acquisition_channel", "location_region", "income_level",
        "recency_tier", "discount_tier", "product_breadth",
        "expected_clv_at_risk", "est_refund_revenue_at_risk",
        "customer_satisfaction_score", "retention_urgency_score", "priority_score",
    ]
    display_cols = [c for c in display_cols if c in d.columns]
    d_show = d[display_cols].round(3).head(500)
    d_show.columns = [c.replace("_", " ").title() for c in d_show.columns]

    count_txt = f"Showing {len(d_show):,} of {len(d):,} filtered customers (capped at 500 rows)"

    table = dash_table.DataTable(
        data=d_show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in d_show.columns],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": SURFACE2, "color": ACCENT,
                      "fontWeight": "700", "border": "1px solid #1e3a5f"},
        style_cell={"backgroundColor": SURFACE, "color": FONT,
                    "border": "1px solid #1e3a5f", "fontSize": "0.78rem",
                    "textAlign": "left", "padding": "7px 10px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#0f2038"},
        ],
    )
    return count_txt, table


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8050"))
    print("=" * 60)
    print("  CLV Business Intelligence Dashboard")
    print(f"  Open: http://localhost:{port}")
    print("=" * 60)
    app.run(debug=False, host=host, port=port)
