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


PLOTLY_CONFIG = getattr(
    dash_app,
    "INTERACTIVE_GRAPH_CONFIG",
    {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    },
)


def _theme(fig):
    fig.update_layout(**dash_app.LAYOUT_BASE)
    return fig


def _format_inr(value, decimals=0):
    return f"Rs {value:,.{decimals}f}"


def _page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-banner">
            <div class="page-kicker">Big Data CLV Analytics</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight_box(lines, accent=None):
    accent = accent or dash_app.BLUE
    if isinstance(lines, str):
        lines = [lines]
    items = "".join(f"<li>{line}</li>" for line in lines if line)
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: {accent};">
            <div class="insight-title">Business takeaway</div>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chart(fig, insights, accent=None):
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    _insight_box(insights, accent=accent)


def _filtered_controls(prefix, return_state=False):
    c1, c2, c3, c4 = st.columns(4)
    clv = c1.selectbox(
        "CLV Segment",
        ["All", "High Value", "Medium Value", "Low Value"],
        key=f"{prefix}_clv",
    )
    churn = c2.selectbox(
        "Churn Risk",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        key=f"{prefix}_churn",
    )
    channel = c3.selectbox(
        "Channel",
        ["All"] + dash_app.ALL_CHANNELS,
        key=f"{prefix}_channel",
    )
    region = c4.selectbox(
        "Region",
        ["All"] + dash_app.ALL_REGIONS,
        key=f"{prefix}_region",
    )
    filtered = dash_app.filter_dataset(
        dash_app.df,
        clv=clv,
        churn=churn,
        channel=channel,
        region=region,
    )
    state = {
        "clv": clv,
        "churn": churn,
        "channel": channel,
        "region": region,
    }
    if return_state:
        return filtered, state
    return filtered


def _overview_page():
    _page_header(
        "CLV Intelligence Dashboard",
        "Filter the customer base, compare value segments, and turn churn signals into concrete action.",
    )
    d = _filtered_controls("ov")
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    danger = d[(d["clv_segment"] == "High Value") & (d["churn_risk"] == "High Risk")]
    best_channel = d.groupby("acquisition_channel")["Customer_Lifetime_Value"].mean().idxmax()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Customers", f"{len(d):,}")
    m2.metric("Avg CLV", _format_inr(d["Customer_Lifetime_Value"].mean()))
    m3.metric("Danger Zone", f"{len(danger):,}")
    m4.metric("Avg Churn", f"{d['churn_probability'].mean():.1%}")
    m5.metric("Best Channel", best_channel)

    seg_cnt = d["clv_segment"].value_counts()
    pie = go.Figure(go.Pie(labels=seg_cnt.index, values=seg_cnt.values, hole=0.45))
    _theme(pie).update_layout(title="CLV Segment Distribution")

    risk_cnt = d["churn_risk"].value_counts().reindex(dash_app.RISK_ORDER).dropna()
    risk_bar = go.Figure(
        go.Bar(
            x=risk_cnt.index,
            y=risk_cnt.values,
            marker_color=[dash_app.RED, dash_app.YELLOW, dash_app.GREEN],
            text=risk_cnt.values,
            texttemplate="%{text:,}",
            textposition="outside",
        )
    )
    _theme(risk_bar).update_layout(title="Churn Risk Count")

    rev_cnt = d["revenue_band"].value_counts().reindex(dash_app.REVENUE_ORDER).dropna()
    rev_colors = {
        "Platinum": dash_app.PURPLE,
        "Gold": dash_app.YELLOW,
        "Silver": dash_app.BLUE,
        "Bronze": dash_app.MUTED,
    }
    revenue_donut = go.Figure(
        go.Pie(
            labels=rev_cnt.index,
            values=rev_cnt.values,
            hole=0.45,
            marker=dict(
                colors=[rev_colors.get(label, dash_app.CYAN) for label in rev_cnt.index],
                line=dict(color=dash_app.BG, width=3),
            ),
            textinfo="label+percent",
        )
    )
    _theme(revenue_donut).update_layout(title="Revenue Band Composition")

    sample = d.sample(min(2500, len(d)), random_state=42)
    scatter = px.scatter(
        sample,
        x="churn_probability",
        y="Customer_Lifetime_Value",
        color="clv_segment",
        size="engagement_score_normalized",
        color_discrete_map={
            "High Value": dash_app.BLUE,
            "Medium Value": dash_app.GREEN,
            "Low Value": dash_app.RED,
        },
        labels={
            "churn_probability": "Churn Probability",
            "Customer_Lifetime_Value": "CLV (Rs)",
        },
    )
    scatter.add_vrect(x0=0.65, x1=1.0, fillcolor=dash_app.RED, opacity=0.08, line_width=0)
    _theme(scatter).update_layout(title="CLV vs Churn Probability")

    top_segment = seg_cnt.idxmax()
    top_segment_share = seg_cnt.max() / len(d)
    platinum_share = d["revenue_band"].eq("Platinum").mean()

    c1, c2 = st.columns(2)
    with c1:
        _render_chart(
            pie,
            [
                f"{top_segment} customers make up {top_segment_share:.1%} of the filtered cohort.",
                "Move medium-value accounts into premium journeys first because that segment usually gives the biggest scalable CLV lift.",
            ],
            accent=dash_app.BLUE,
        )
    with c2:
        _render_chart(
            risk_bar,
            [
                f"{risk_cnt.get('High Risk', 0):,} customers currently sit in the high-risk bucket.",
                "Prioritize retention outreach on that cohort before expanding acquisition spend, because recovered value is cheaper than replacing churned revenue.",
            ],
            accent=dash_app.RED,
        )

    c3, c4 = st.columns(2)
    with c3:
        _render_chart(
            revenue_donut,
            [
                f"Platinum customers are only {platinum_share:.1%} of the base, but they represent the premium layer of the portfolio.",
                "Protect this cohort with white-glove service and exclusive offers rather than broad discounts.",
            ],
            accent=dash_app.PURPLE,
        )
    with c4:
        _render_chart(
            scatter,
            [
                f"The danger zone contains {len(danger):,} high-value customers with average CLV of {_format_inr(danger['Customer_Lifetime_Value'].mean() if len(danger) else 0)}.",
                "Use this plot to isolate customers with both high value and high churn, then route them into fast, personalized save campaigns.",
            ],
            accent=dash_app.YELLOW,
        )


def _channel_page():
    _page_header(
        "Channel & Revenue Intel",
        "Compare which acquisition channels create the healthiest mix of CLV, revenue quality, and retention headroom.",
    )
    c1, c2 = st.columns(2)
    clv = c1.selectbox(
        "CLV Segment",
        ["All", "High Value", "Medium Value", "Low Value"],
        key="channel_clv",
    )
    churn = c2.selectbox(
        "Churn Risk",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        key="channel_churn",
    )
    d = dash_app.filter_dataset(dash_app.df, clv=clv, churn=churn)
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    channel_agg = d.groupby("acquisition_channel").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_revenue=("total_revenue_generated", "mean"),
        avg_churn=("churn_probability", "mean"),
        customers=("Customer_Lifetime_Value", "count"),
    )
    top_clv = channel_agg["avg_clv"].idxmax()
    top_revenue = channel_agg["avg_revenue"].idxmax()
    lowest_churn = channel_agg["avg_churn"].idxmin()
    largest = channel_agg["customers"].idxmax()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best CLV Channel", top_clv, _format_inr(channel_agg.loc[top_clv, "avg_clv"]))
    m2.metric("Revenue Leader", top_revenue, _format_inr(channel_agg.loc[top_revenue, "avg_revenue"]))
    m3.metric("Lowest Churn", lowest_churn, f"{channel_agg.loc[lowest_churn, 'avg_churn']:.1%}")
    m4.metric("Largest Channel", largest, f"{channel_agg.loc[largest, 'customers']:,.0f}")

    premium_mix = (
        d.assign(is_premium=d["revenue_band"].isin(["Gold", "Platinum"]))
        .groupby("acquisition_channel")["is_premium"]
        .mean()
    )
    premium_leader = premium_mix.idxmax()
    engine_channel = (
        channel_agg.assign(value_index=channel_agg["avg_revenue"] * channel_agg["customers"])
        ["value_index"]
        .idxmax()
    )

    bar, stacked, bubble = dash_app.cb_channel(clv, churn)
    c3, c4 = st.columns(2)
    with c3:
        _render_chart(
            bar,
            [
                f"{top_clv} leads channel CLV at {_format_inr(channel_agg.loc[top_clv, 'avg_clv'])}.",
                "Shift spend toward the acquisition motions used by that channel, then test how much of its onboarding quality can be replicated elsewhere.",
            ],
            accent=dash_app.BLUE,
        )
    with c4:
        _render_chart(
            stacked,
            [
                f"{premium_leader} has the strongest Gold/Platinum mix at {premium_mix.loc[premium_leader]:.1%}.",
                "Budget decisions should favor channels that bring premium customers, not just raw volume.",
            ],
            accent=dash_app.PURPLE,
        )

    _render_chart(
        bubble,
        [
            f"{engine_channel} is currently the biggest commercial engine when volume and revenue quality are combined.",
            "Use the top-right channels as retention and upsell priorities because repeat orders there compound faster than broad acquisition alone.",
        ],
        accent=dash_app.CYAN,
    )


def _product_page():
    _page_header(
        "Product & Category Insights",
        "Use category performance to decide where to place premium inventory, cross-sell effort, and discount pressure.",
    )
    income = st.selectbox("Income Level", ["All", "High", "Medium", "Low"])
    d = dash_app.df if income == "All" else dash_app.df[dash_app.df["income_level"] == income]
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    cat = d.groupby("product_category_preference").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_rev=("total_revenue_generated", "mean"),
        avg_repeat=("repeat_purchase_rate", "mean"),
        avg_return=("return_refund_rate", "mean"),
        avg_disc=("discount_sensitivity", "mean"),
    ).reset_index()

    top_clv_row = cat.loc[cat["avg_clv"].idxmax()]
    top_rev_row = cat.loc[cat["avg_rev"].idxmax()]
    top_repeat_row = cat.loc[cat["avg_repeat"].idxmax()]
    low_return_row = cat.loc[cat["avg_return"].idxmin()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top CLV Category", top_clv_row["product_category_preference"])
    m2.metric("Revenue Leader", top_rev_row["product_category_preference"])
    m3.metric("Highest Repeat Rate", f"{top_repeat_row['avg_repeat']:.1%}")
    m4.metric("Lowest Return Rate", f"{low_return_row['avg_return']:.1%}")

    _, clv_bar, rev_bar, scatter = dash_app.cb_product(income)
    c1, c2 = st.columns(2)
    with c1:
        _render_chart(
            clv_bar,
            [
                f"{top_clv_row['product_category_preference']} leads CLV at {_format_inr(top_clv_row['avg_clv'])}.",
                "Give the highest-value categories better visibility, premium bundles, and loyalty perks before pushing blanket markdowns.",
            ],
            accent=dash_app.BLUE,
        )
    with c2:
        _render_chart(
            rev_bar,
            [
                f"{top_rev_row['product_category_preference']} is the near-term revenue leader at {_format_inr(top_rev_row['avg_rev'])}.",
                "When CLV and revenue leaders differ, use the gap to decide where to chase short-term sales versus long-term relationship growth.",
            ],
            accent=dash_app.GREEN,
        )

    _render_chart(
        scatter,
        [
            f"{top_repeat_row['product_category_preference']} has the highest repeat rate at {top_repeat_row['avg_repeat']:.1%}, while {low_return_row['product_category_preference']} keeps return leakage lowest.",
            "Favor categories with strong repeat behavior and low discount dependence for bundles, memberships, and cross-sell campaigns.",
        ],
        accent=dash_app.YELLOW,
    )


def _churn_page():
    _page_header(
        "Churn & Retention Planner",
        "Surface the customers most likely to defect, then rank where the retention team should intervene first.",
    )
    top_n = st.slider("Top N Danger Zone Customers", 5, 50, 20, 5)
    violin, heat, _ = dash_app.cb_churn(top_n)

    danger = dash_app.df[
        (dash_app.df["clv_segment"] == "High Value")
        & (dash_app.df["churn_risk"] == "High Risk")
    ].copy()
    hv_total = dash_app.df["clv_segment"].eq("High Value").sum()
    hv_risky_share = len(danger) / hv_total if hv_total else 0

    c1, c2 = st.columns(2)
    with c1:
        _render_chart(
            violin,
            [
                f"{hv_risky_share:.1%} of high-value customers already fall into the high-risk churn bucket.",
                "Target the upper tail of this distribution first because even small save-rate gains here preserve outsized future revenue.",
            ],
            accent=dash_app.RED,
        )
    with c2:
        _render_chart(
            heat,
            [
                f"The High Value x High Risk cell currently holds {len(danger):,} customers.",
                "Treat that cell as the first retention queue and assign fast outreach, service recovery, or tailored offers before scaling broader campaigns.",
            ],
            accent=dash_app.YELLOW,
        )

    cols = [
        "Customer_Lifetime_Value",
        "churn_probability",
        "engagement_score_normalized",
        "customer_satisfaction_score",
        "acquisition_channel",
        "location_region",
        "retention_urgency_score",
        "priority_score",
    ]
    st.dataframe(
        danger[cols].sort_values("retention_urgency_score", ascending=False).head(top_n).round(3),
        use_container_width=True,
    )
    top_urgency = danger["retention_urgency_score"].max() if len(danger) else 0
    _insight_box(
        [
            f"The current top account on the retention queue has an urgency score of {top_urgency:,.1f}.",
            "Work this table from the top down and keep response SLAs tight, because delay matters most for high-value accounts already signaling exit intent.",
        ],
        accent=dash_app.RED,
    )


def _regional_page():
    _page_header(
        "Regional & Demographic Analysis",
        "Compare where value is concentrated geographically and which demographic pockets deserve tailored growth or retention playbooks.",
    )
    region_agg = dash_app.df.groupby("location_region").agg(
        avg_clv=("Customer_Lifetime_Value", "mean"),
        avg_churn=("churn_probability", "mean"),
        avg_satisfaction=("customer_satisfaction_score", "mean"),
        customers=("Customer_Lifetime_Value", "count"),
    )
    top_region = region_agg["avg_clv"].idxmax()
    stable_region = region_agg["avg_churn"].idxmin()
    happy_region = region_agg["avg_satisfaction"].idxmax()
    largest_region = region_agg["customers"].idxmax()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top CLV Region", top_region, _format_inr(region_agg.loc[top_region, "avg_clv"]))
    m2.metric("Lowest Churn Region", stable_region, f"{region_agg.loc[stable_region, 'avg_churn']:.1%}")
    m3.metric("Highest Satisfaction", happy_region, f"{region_agg.loc[happy_region, 'avg_satisfaction']:.2f}")
    m4.metric("Largest Region", largest_region, f"{region_agg.loc[largest_region, 'customers']:,.0f}")

    dual, box, gender, satisfaction = dash_app.cb_regional("/regional")
    income_medians = (
        dash_app.df.groupby("income_level")["Customer_Lifetime_Value"].median().sort_values(ascending=False)
    )
    gen_grp = (
        dash_app.df.groupby(["clv_segment", "gender"]).size().reset_index(name="count")
        if "gender" in dash_app.df.columns
        else None
    )
    top_gender_row = gen_grp.loc[gen_grp["count"].idxmax()] if gen_grp is not None and not gen_grp.empty else None
    sat_cell = (
        dash_app.df.groupby(["location_region", "churn_risk"])["customer_satisfaction_score"]
        .mean()
        .reset_index()
        .sort_values("customer_satisfaction_score")
        .iloc[0]
    )

    c1, c2 = st.columns(2)
    with c1:
        region_churn_vs_avg = region_agg.loc[top_region, "avg_churn"] - region_agg["avg_churn"].mean()
        direction = "above" if region_churn_vs_avg > 0 else "below"
        _render_chart(
            dual,
            [
                f"{top_region} leads regional CLV at {_format_inr(region_agg.loc[top_region, 'avg_clv'])}, with churn {abs(region_churn_vs_avg):.1%} {direction} the regional average.",
                "Regions that combine high CLV with elevated churn should get localized retention budgets before expanding to lower-value markets.",
            ],
            accent=dash_app.BLUE,
        )
    with c2:
        _render_chart(
            box,
            [
                f"{income_medians.index[0]} income customers show the highest median CLV at {_format_inr(income_medians.iloc[0])}.",
                "Use income-specific product bundles and service tiers instead of one-size-fits-all offers, because upside varies meaningfully across segments.",
            ],
            accent=dash_app.GREEN,
        )

    c3, c4 = st.columns(2)
    with c3:
        gender_lines = [
            f"{top_gender_row['gender']} customers are most concentrated in the {top_gender_row['clv_segment']} segment."
            if top_gender_row is not None
            else "Gender coverage is limited in the current dataset."
        ]
        gender_lines.append(
            "Use this split to test more precise messaging and merchandising by high-value audience rather than spending evenly across broad campaigns."
        )
        _render_chart(gender, gender_lines, accent=dash_app.PURPLE)
    with c4:
        _render_chart(
            satisfaction,
            [
                f"{sat_cell['location_region']} / {sat_cell['churn_risk']} is the weakest satisfaction pocket at {sat_cell['customer_satisfaction_score']:.2f}.",
                "Improve service recovery and feedback loops first in low-satisfaction, high-risk regions because they compound value leakage fastest.",
            ],
            accent=dash_app.RED,
        )


def _engagement_page():
    _page_header(
        "Engagement & Loyalty Analysis",
        "See how attention, membership, and service friction shape long-term customer value.",
    )
    loyalty = dash_app.df.groupby("loyalty_label")["Customer_Lifetime_Value"].mean()
    member_clv = loyalty.get("Member", 0)
    non_member_clv = loyalty.get("Non-Member", 0)
    lift = member_clv - non_member_clv
    member_share = dash_app.df["loyalty_label"].eq("Member").mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Engagement", f"{dash_app.df['engagement_score_normalized'].mean():.1f}/100")
    m2.metric("Loyalty CLV Lift", _format_inr(lift))
    m3.metric("Member Share", f"{member_share:.1%}")
    m4.metric("Avg Support Tickets", f"{dash_app.df['customer_support_tickets'].mean():.2f}")

    hist, scatter, loyalty_fig, support_fig = dash_app.cb_engagement("/engagement")
    segment_engagement = (
        dash_app.df.groupby("clv_segment")["engagement_score_normalized"].mean().sort_values(ascending=False)
    )
    open_rate_split = (
        dash_app.df.groupby("loyalty_label")["email_open_rate"].mean().sort_values(ascending=False)
    )
    support_split = (
        dash_app.df.groupby("clv_segment")["customer_support_tickets"].mean().sort_values(ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        _render_chart(
            hist,
            [
                f"{segment_engagement.index[0]} customers show the strongest average engagement at {segment_engagement.iloc[0]:.1f}/100.",
                "Invest in the engagement motions that already work for your highest-value segment, then adapt them for medium-value cohorts to lift CLV efficiently.",
            ],
            accent=dash_app.BLUE,
        )
    with c2:
        _render_chart(
            scatter,
            [
                f"{open_rate_split.index[0]} customers open email most often at {open_rate_split.iloc[0]:.1%}.",
                "Treat email and loyalty as linked levers: stronger nurture sequences work best when they move customers into membership or tier progression.",
            ],
            accent=dash_app.CYAN,
        )

    c3, c4 = st.columns(2)
    with c3:
        _render_chart(
            loyalty_fig,
            [
                f"Loyalty membership lifts average CLV by {_format_inr(lift)}.",
                "Reduce signup friction and promote benefits earlier in the customer journey, because membership is already associated with materially better value.",
            ],
            accent=dash_app.PURPLE,
        )
    with c4:
        _render_chart(
            support_fig,
            [
                f"{support_split.index[0]} customers generate the most support demand at {support_split.iloc[0]:.2f} tickets on average.",
                "If high-value segments keep surfacing in support, route those issues into product and CX fixes quickly so service cost does not erode CLV.",
            ],
            accent=dash_app.RED,
        )


def _risk_lab_page():
    _page_header(
        "Big Data Risk Lab",
        "Turn the engineered recency, refund, discount, and breadth signals into a working decision cockpit for the business.",
    )
    d, filters = _filtered_controls("risk", return_state=True)
    if d.empty:
        st.warning("No data for the selected filters.")
        return

    lens = st.selectbox(
        "Business Lens",
        [
            "recency_tier",
            "discount_tier",
            "product_breadth",
            "revenue_band",
            "acquisition_channel",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
    )

    high_risk = d[d["churn_risk"] == "High Risk"]
    danger = d[(d["clv_segment"] == "High Value") & (d["churn_risk"] == "High Risk")]
    dormant_hv = d[
        (d["clv_segment"] == "High Value")
        & (d["recency_tier"] == "Dormant (>90 days)")
    ]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Filtered Cohort", f"{len(d):,}")
    m2.metric("Expected CLV at Risk", _format_inr(high_risk["expected_clv_at_risk"].sum()))
    m3.metric("Danger Zone Value", _format_inr(danger["Customer_Lifetime_Value"].sum()))
    m4.metric("Dormant High Value", f"{len(dormant_hv):,}")
    m5.metric("Refund Exposure", _format_inr(d["est_refund_revenue_at_risk"].sum()))

    _, rfm, returns, discount, breadth, lens_fig, pipeline = dash_app.cb_risk_lab(
        filters["clv"],
        filters["churn"],
        filters["channel"],
        filters["region"],
        lens,
    )

    refund_cell = (
        d.groupby(["clv_segment", "churn_risk"])["est_refund_revenue_at_risk"]
        .mean()
        .reset_index()
        .sort_values("est_refund_revenue_at_risk", ascending=False)
        .iloc[0]
    )
    discount_agg = (
        d.groupby("discount_tier")[["Customer_Lifetime_Value", "return_refund_rate"]]
        .mean()
        .reindex(dash_app.DISCOUNT_ORDER)
        .dropna(how="all")
    )
    breadth_agg = (
        d.groupby("product_breadth")[["Customer_Lifetime_Value", "churn_probability"]]
        .mean()
        .reindex(dash_app.BREADTH_ORDER)
        .dropna(how="all")
    )
    lens_agg = d.groupby(lens)["churn_probability"].mean().sort_values(ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        _render_chart(
            rfm,
            [
                f"There are {len(dormant_hv):,} dormant high-value customers in the current cohort.",
                "Win-back campaigns should start there because those customers have proven value but weakening recency signals.",
            ],
            accent=dash_app.BLUE,
        )
    with c2:
        _render_chart(
            returns,
            [
                f"{refund_cell['clv_segment']} / {refund_cell['churn_risk']} has the highest average refund exposure at {_format_inr(refund_cell['est_refund_revenue_at_risk'], 1)}.",
                "Treat that cell as a product-fit and service-quality investigation queue, not just a retention issue.",
            ],
            accent=dash_app.RED,
        )

    c3, c4 = st.columns(2)
    with c3:
        top_discount_clv = discount_agg["Customer_Lifetime_Value"].idxmax()
        top_return_tier = discount_agg["return_refund_rate"].idxmax()
        _render_chart(
            discount,
            [
                f"{top_discount_clv} customers hold the best average CLV, while {top_return_tier} customers carry the highest return pressure.",
                "Keep broad discounts away from weak-quality cohorts and shift value messaging toward bundles, exclusivity, or loyalty benefits.",
            ],
            accent=dash_app.YELLOW,
        )
    with c4:
        high_breadth = breadth_agg["Customer_Lifetime_Value"].idxmax()
        low_churn_breadth = breadth_agg["churn_probability"].idxmin()
        _render_chart(
            breadth,
            [
                f"{high_breadth} customers deliver the strongest CLV, and {low_churn_breadth} customers are the stickiest.",
                "Cross-sell medium-breadth customers into adjacent categories because breadth is acting like a loyalty multiplier here.",
            ],
            accent=dash_app.CYAN,
        )

    c5, c6 = st.columns(2)
    with c5:
        hotspot = lens_agg.index[0]
        _render_chart(
            lens_fig,
            [
                f"Within the current {lens.replace('_', ' ')} view, {hotspot} has the highest average churn at {lens_agg.iloc[0]:.1%}.",
                "Use the lens switcher to decide whether the next decision should be about recency, discount behavior, category breadth, revenue tier, or channel mix.",
            ],
            accent=dash_app.PURPLE,
        )
    with c6:
        _render_chart(
            pipeline,
            [
                f"The dashboard is currently visualizing {len(d):,} rows out of {len(dash_app.df):,} engineered customer records.",
                "That filtered pipeline makes it easy to move from broad analytics to operational action without losing the business context behind the numbers.",
            ],
            accent=dash_app.GREEN,
        )


def _explorer_page():
    _page_header(
        "Data Explorer",
        "Inspect the engineered dataset directly, validate cohort assumptions, and hand off grounded evidence to downstream teams.",
    )
    d = _filtered_controls("exp")
    band = st.selectbox("Revenue Band", ["All", "Platinum", "Gold", "Silver", "Bronze"])
    if band != "All":
        d = d[d["revenue_band"] == band]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Rows", f"{len(d):,}")
    m2.metric("Avg CLV", _format_inr(d["Customer_Lifetime_Value"].mean() if len(d) else 0))
    m3.metric("Avg Churn", f"{d['churn_probability'].mean():.1%}" if len(d) else "0.0%")
    m4.metric("Avg Revenue", _format_inr(d["total_revenue_generated"].mean() if len(d) else 0))

    cols = [
        "clv_segment",
        "churn_risk",
        "revenue_band",
        "Customer_Lifetime_Value",
        "churn_probability",
        "total_revenue_generated",
        "purchase_frequency",
        "average_order_value",
        "engagement_score_normalized",
        "acquisition_channel",
        "location_region",
        "income_level",
        "recency_tier",
        "discount_tier",
        "product_breadth",
        "expected_clv_at_risk",
        "est_refund_revenue_at_risk",
        "customer_satisfaction_score",
        "retention_urgency_score",
        "priority_score",
    ]
    st.caption(f"Showing {min(len(d), 1000):,} of {len(d):,} filtered customers")
    st.dataframe(d[cols].round(3).head(1000), use_container_width=True)
    _insight_box(
        [
            "Use the filters to isolate a business cohort, then sort the table by urgency, value, or risk to validate what the charts are signaling.",
            "This page works best as the handoff layer between executive views and analyst or operations follow-through.",
        ],
        accent=dash_app.BLUE,
    )


def render_app():
    st.set_page_config(page_title="CLV Intelligence Dashboard", page_icon="📊", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {dash_app.BG};
            color: {dash_app.FONT};
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
        [data-testid="stSidebar"] {{
            background: {dash_app.SURFACE};
        }}
        div[data-testid="stMetric"] {{
            background: {dash_app.SURFACE2};
            border: 1px solid #1e3a5f;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        }}
        .page-banner {{
            background: linear-gradient(135deg, #10213a 0%, #13273c 58%, #123333 100%);
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 16px;
        }}
        .page-banner h1 {{
            margin: 0;
            color: #f8fafc;
            font-size: 1.85rem;
            line-height: 1.15;
        }}
        .page-banner p {{
            margin: 8px 0 0;
            color: #cbd5e1;
            font-size: 0.96rem;
            line-height: 1.5;
        }}
        .page-kicker {{
            color: {dash_app.ACCENT};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .insight-card {{
            background: #0f2038;
            border-left: 4px solid {dash_app.BLUE};
            border-radius: 0 8px 8px 0;
            padding: 12px 14px;
            margin: 4px 0 18px;
        }}
        .insight-title {{
            color: #ffffff;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }}
        .insight-card ul {{
            margin: 0;
            padding-left: 18px;
            color: #cbd5e1;
        }}
        .insight-card li {{
            margin: 0 0 4px;
            line-height: 1.45;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("CLV Intel")
    page_options = [
        "Executive Overview",
        "Channel & Revenue Intel",
        "Churn & Retention",
        "Product & Category",
        "Regional & Demographics",
        "Engagement & Loyalty",
        "Big Data Risk Lab",
        "Data Explorer",
    ]
    page_options = [page for page in page_options if page != "Big Data Risk Lab"]
    page = st.sidebar.radio("Dashboard page", page_options)

    if page == "Executive Overview":
        _overview_page()
    elif page == "Channel & Revenue Intel":
        _channel_page()
    elif page == "Product & Category":
        _product_page()
    elif page == "Churn & Retention":
        _churn_page()
    elif page == "Regional & Demographics":
        _regional_page()
    elif page == "Engagement & Loyalty":
        _engagement_page()
    else:
        _explorer_page()


if __name__ == "__main__":
    render_app()
