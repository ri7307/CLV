"""Streamlit Cloud entrypoint for the CLV dashboard.

Use this file as the main file on Streamlit Cloud:
    streamlit_app.py
"""

import sys
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_main_module = sys.modules.get("__main__")
if hasattr(_main_module, "df") and hasattr(_main_module, "cb_product"):
    dash_app = _main_module
else:
    import clv_dashboard as dash_app


def _theme(fig):
    fig.update_layout(**dash_app.LAYOUT_BASE)
    return fig


def _filtered_controls(prefix):
    c1, c2, c3, c4 = st.columns(4)
    clv = c1.selectbox(
        "CLV Segment", ["All", "High Value", "Medium Value", "Low Value"],
        key=f"{prefix}_clv",
    )
    churn = c2.selectbox(
        "Churn Risk", ["All", "High Risk", "Medium Risk", "Low Risk"],
        key=f"{prefix}_churn",
    )
    channel = c3.selectbox("Channel", ["All"] + dash_app.ALL_CHANNELS,
                           key=f"{prefix}_channel")
    region = c4.selectbox("Region", ["All"] + dash_app.ALL_REGIONS,
                          key=f"{prefix}_region")
    return dash_app.filter_dataset(
        dash_app.df, clv=clv, churn=churn, channel=channel, region=region,
    )


def _overview_page():
    st.title("CLV Intelligence Dashboard")
    d = _filtered_controls("ov")
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    danger = d[(d["clv_segment"] == "High Value") & (d["churn_risk"] == "High Risk")]
    best_channel = d.groupby("acquisition_channel")["Customer_Lifetime_Value"].mean().idxmax()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Customers", f"{len(d):,}")
    m2.metric("Avg CLV", f"₹{d['Customer_Lifetime_Value'].mean():,.0f}")
    m3.metric("Danger Zone", f"{len(danger):,}")
    m4.metric("Avg Churn", f"{d['churn_probability'].mean():.1%}")
    m5.metric("Best Channel", best_channel)

    seg_cnt = d["clv_segment"].value_counts()
    pie = go.Figure(go.Pie(labels=seg_cnt.index, values=seg_cnt.values, hole=0.45))
    _theme(pie).update_layout(title="CLV Segment Distribution")

    risk_cnt = d["churn_risk"].value_counts().reindex(dash_app.RISK_ORDER).dropna()
    risk_bar = go.Figure(go.Bar(
        x=risk_cnt.index, y=risk_cnt.values,
        marker_color=[dash_app.RED, dash_app.YELLOW, dash_app.GREEN],
        text=risk_cnt.values, texttemplate="%{text:,}", textposition="outside",
    ))
    _theme(risk_bar).update_layout(title="Churn Risk Count")

    sample = d.sample(min(2500, len(d)), random_state=42)
    scatter = px.scatter(
        sample, x="churn_probability", y="Customer_Lifetime_Value",
        color="clv_segment", size="engagement_score_normalized",
        color_discrete_map={
            "High Value": dash_app.BLUE,
            "Medium Value": dash_app.GREEN,
            "Low Value": dash_app.RED,
        },
        labels={"churn_probability": "Churn Probability",
                "Customer_Lifetime_Value": "CLV (₹)"},
    )
    scatter.add_vrect(x0=0.65, x1=1.0, fillcolor=dash_app.RED, opacity=0.08,
                      line_width=0)
    _theme(scatter).update_layout(title="CLV vs Churn Probability")

    c1, c2 = st.columns(2)
    c1.plotly_chart(pie, use_container_width=True)
    c2.plotly_chart(risk_bar, use_container_width=True)
    st.plotly_chart(scatter, use_container_width=True)


def _product_page():
    st.title("Product & Category Insights")
    income = st.selectbox("Income Level", ["All", "High", "Medium", "Low"])
    d = dash_app.df if income == "All" else dash_app.df[dash_app.df["income_level"] == income]
    cat = d.groupby("product_category_preference").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_rev=("total_revenue_generated", "mean"),
        avg_repeat=("repeat_purchase_rate", "mean"),
        avg_return=("return_refund_rate", "mean"),
    ).reset_index()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top CLV Category", cat.loc[cat["avg_clv"].idxmax(), "product_category_preference"])
    m2.metric("Revenue Leader", cat.loc[cat["avg_rev"].idxmax(), "product_category_preference"])
    m3.metric("Highest Repeat Rate", f"{cat['avg_repeat'].max():.1%}")
    m4.metric("Lowest Return Rate", f"{cat['avg_return'].min():.1%}")

    _, clv_bar, rev_bar, scatter = dash_app.cb_product(income)
    c1, c2 = st.columns(2)
    c1.plotly_chart(clv_bar, use_container_width=True)
    c2.plotly_chart(rev_bar, use_container_width=True)
    st.plotly_chart(scatter, use_container_width=True)


def _churn_page():
    st.title("Churn & Retention Planner")
    top_n = st.slider("Top N Danger Zone Customers", 5, 50, 20, 5)
    violin, heat, _ = dash_app.cb_churn(top_n)
    c1, c2 = st.columns(2)
    c1.plotly_chart(violin, use_container_width=True)
    c2.plotly_chart(heat, use_container_width=True)

    danger = dash_app.df[
        (dash_app.df["clv_segment"] == "High Value") &
        (dash_app.df["churn_risk"] == "High Risk")
    ]
    cols = [
        "Customer_Lifetime_Value", "churn_probability",
        "engagement_score_normalized", "customer_satisfaction_score",
        "acquisition_channel", "location_region",
        "retention_urgency_score", "priority_score",
    ]
    st.dataframe(
        danger[cols].sort_values("retention_urgency_score", ascending=False)
        .head(top_n).round(3),
        use_container_width=True,
    )


def _risk_lab_page():
    st.title("Big Data Risk Lab")
    d = _filtered_controls("risk")
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    high_risk = d[d["churn_risk"] == "High Risk"]
    danger = d[(d["clv_segment"] == "High Value") & (d["churn_risk"] == "High Risk")]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Cohort", f"{len(d):,}")
    m2.metric("Expected CLV at Risk", f"₹{high_risk['expected_clv_at_risk'].sum():,.0f}")
    m3.metric("Danger Zone Value", f"₹{danger['Customer_Lifetime_Value'].sum():,.0f}")
    m4.metric("Refund Exposure", f"₹{d['est_refund_revenue_at_risk'].sum():,.0f}")

    _, rfm, returns, discount, breadth, lens, pipeline = dash_app.cb_risk_lab(
        "All", "All", "All", "All", "recency_tier",
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(rfm, use_container_width=True)
    c2.plotly_chart(returns, use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(discount, use_container_width=True)
    c4.plotly_chart(breadth, use_container_width=True)
    c5, c6 = st.columns(2)
    c5.plotly_chart(lens, use_container_width=True)
    c6.plotly_chart(pipeline, use_container_width=True)


def _explorer_page():
    st.title("Data Explorer")
    d = _filtered_controls("exp")
    band = st.selectbox("Revenue Band", ["All", "Platinum", "Gold", "Silver", "Bronze"])
    if band != "All":
        d = d[d["revenue_band"] == band]

    cols = [
        "clv_segment", "churn_risk", "revenue_band", "Customer_Lifetime_Value",
        "churn_probability", "total_revenue_generated", "purchase_frequency",
        "average_order_value", "engagement_score_normalized", "acquisition_channel",
        "location_region", "income_level", "recency_tier", "discount_tier",
        "product_breadth", "expected_clv_at_risk", "est_refund_revenue_at_risk",
        "customer_satisfaction_score", "retention_urgency_score", "priority_score",
    ]
    st.caption(f"Showing {min(len(d), 1000):,} of {len(d):,} filtered customers")
    st.dataframe(d[cols].round(3).head(1000), use_container_width=True)


def render_app():
    st.set_page_config(page_title="CLV Intelligence Dashboard", page_icon="📊",
                       layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {dash_app.BG}; color: {dash_app.FONT}; }}
        [data-testid="stSidebar"] {{ background: {dash_app.SURFACE}; }}
        div[data-testid="stMetric"] {{
            background: {dash_app.SURFACE2};
            border: 1px solid #1e3a5f;
            border-radius: 8px;
            padding: 12px 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("CLV Intel")
    page = st.sidebar.radio(
        "Dashboard page",
        ["Executive Overview", "Product & Category", "Churn & Retention",
         "Big Data Risk Lab", "Data Explorer"],
    )

    if page == "Executive Overview":
        _overview_page()
    elif page == "Product & Category":
        _product_page()
    elif page == "Churn & Retention":
        _churn_page()
    elif page == "Big Data Risk Lab":
        _risk_lab_page()
    else:
        _explorer_page()


if __name__ == "__main__":
    render_app()
