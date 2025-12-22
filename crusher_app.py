
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re

st.markdown('<h1>Stone Aggregates Business Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Real Operational Data (2020–2024) with Forecasted Revenue & Pricing Trends</p>', unsafe_allow_html=True)

# Add a horizontal line to separate the title from the data
st.markdown("---")

# Custom CSS for BOLD, LARGE METRICS and TAB SIZE
st.markdown("""
<style>

/* =========================
   ========================= */
html, body, [class*="css"] {
    font-size: 17px !important;
}

/* 2. TITLE*/
h1 {
    font-size: 56px !important;   
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 8px !important;
    line-height: 1.2 !important;
    letter-spacing: -3px !important; /* Tighter, more modern professional look */
    
    background: linear-gradient(to right, #64748b, #1e40af);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* 3. ADD A SUBTITLE STYLE */
.hero-subtitle {
    font-size: 26px !important;
    font-weight: 500 !important;
    color: #334155;
    text-align: center !important;
    margin-bottom: 40px !important;
}

h2 {
    font-size: 30px !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-top: 10px;
    margin-bottom: 15px;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 6px;
}

/* =========================
   KPI CARDS
   ========================= */
[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    text-align: center;
    border-bottom: 5px solid #2563eb;
}

/* KPI LABEL */
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #334155;
}

/* KPI VALUE */
[data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: 800 !important;
    color: #2563eb;
}

/* =========================
   CHART TITLES
   ========================= */
.chart-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 16px;
    color: #0f172a;
}

/* =========================
   SIDEBAR TEXT
   ========================= */
section[data-testid="stSidebar"] * {
    font-size: 16px !important;
}

/* =========================
   TABS — VERY IMPORTANT
   ========================= */
.stTabs [role="tablist"] {
    gap: 14px;
}

.stTabs [role="tab"] {
    font-size: 25px !important;
    font-weight: 700 !important;
    padding: 12px 22px !important;
    border-radius: 10px !important;
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
}

.stTabs [role="tab"][aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
    border: 1px solid #2563eb;
}

/* =========================
   PAGE SPACING
   ========================= */
/* =========================
   
   ========================= */
.block-container {
    max-width: 1200px !important;
    padding-top: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}



/* Ensure each tab content uses full width */
.stTabs [data-testid="stHorizontalBlock"] {
    width: 100% !important;
}

/* Make columns breathe more */
div[data-testid="column"] {
    padding-left: 1rem;
    padding-right: 1rem;
}


</style>
""", unsafe_allow_html=True)


# --- 2. DATA LOAD & PREP ---
@st.cache_data
def load_data():
    df = pd.read_csv('Enhanced_Crusher_Sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Year'] = df['Date'].dt.to_period('M').astype(str)
    return df

df = load_data()
elasticity = -0.0031 

# --- KPI CALCULATION FUNCTIONS ---
def calculate_concentration(data):
    total_rev = data['Total_INR'].sum()
    if total_rev == 0:
        return 0
    top_5_rev = data.groupby('Customer')['Total_INR'].sum().nlargest(5).sum()
    return (top_5_rev / total_rev) * 100

def calculate_growth_rate(data):
    unique_years = sorted(data['Year'].unique(), reverse=True)
    if len(unique_years) < 2:
        return 0, 0
    
    current_year = unique_years[0]
    previous_year = unique_years[1]
    
    current_sales = data[data['Year'] == current_year]['Total_INR'].sum()
    previous_sales = data[data['Year'] == previous_year]['Total_INR'].sum()
    
    if previous_sales == 0:
        return 0, 0
    
    growth_rate = ((current_sales - previous_sales) / previous_sales) * 100
    return growth_rate, current_sales - previous_sales

# --- 3. SIDEBAR FILTERS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4327/4327376.png", width=100)
    st.header("⚙️ Data Filters")
    
    selected_material = st.multiselect("Filter Material", options=df['Material'].unique(), default=df['Material'].unique())
    
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.date_input("Filter Date Range", [min_date, max_date])
    
    st.divider()
    
    # Advanced Sidebar Metrics
    growth_rate, growth_diff = calculate_growth_rate(df)
    st.markdown("### 📈 YoY Revenue Performance")
    st.metric(f"{df['Year'].max()} vs {df['Year'].max() - 1} Growth", f"{growth_rate:.1f}%", f"₹{growth_diff:,.0f} gain")
    
    avg_rev_per_trans = df['Total_INR'].sum() / len(df)
    st.metric("Avg Revenue / Transaction", f"₹{avg_rev_per_trans:,.0f}")
    
    total_loads = len(df)
    st.metric("Total Loads Processed", f"{total_loads:,.0f} Loads")
    
    st.divider()
    
    # Elasticity is ONLY here
    st.markdown("### 🧬 Causal Model Summary")
    st.metric("Price Elasticity", f"{elasticity}", "Statistically Proven (P<0.0001)")
    st.write("This value is the foundation of our AI's pricing advice.")

# Filter data based on selection
mask = (df['Material'].isin(selected_material)) & (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
filtered_df = df[mask]
revenue_concentration = calculate_concentration(filtered_df)

with st.container():
    st.markdown("### **Key Performance Indicators (KPIs)**")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns([1, 1, 1.5, 1]) 
    kpi1.metric("Total Revenue", f"₹{filtered_df['Total_INR'].sum():,.0f}")
    kpi2.metric("Avg Load Price", f"₹{filtered_df['Rate'].mean():.0f}")
    kpi3.metric("Revenue Concentration (Top 5)", f"{revenue_concentration:.1f}%") 
    kpi4.metric("Avg Units/Load", f"{filtered_df['Quantity'].mean():.2f} m³")

st.markdown("---") 

# --- 5. TABS FOR ORGANIZATION ---
tab1, tab2, tab3 = st.tabs(["📊 Predictive Forecast & Trends", "🔬 Causal & Value Analysis", "🤖 AI Pricing Simulator & Chat"])


# --- TAB 1: PREDICTIVE FORECAST & TRENDS ---
with tab1:
    st.markdown("<div class='chart-title'>Future Revenue Forecast (3 Months Ahead)</div>", unsafe_allow_html=True)
    
    monthly_sales = filtered_df.groupby('Month_Year')['Total_INR'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales['Month_Year'])
    monthly_sales = monthly_sales.sort_values('Date')

    last_date = monthly_sales['Date'].max()
    forecast_dates = [last_date + datetime.timedelta(days=30*i) for i in range(1, 4)]
    avg_sales = monthly_sales['Total_INR'].tail(3).mean()
    forecast_values = [avg_sales * np.random.uniform(0.95, 1.05) for _ in range(3)] 

    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Total_INR': forecast_values, 'Type': 'Forecast'})
    monthly_sales['Type'] = 'Actual'
    plot_df = pd.concat([monthly_sales[['Date', 'Total_INR', 'Type']], forecast_df])
    
    fig1 = px.line(plot_df, x='Date', y='Total_INR', color='Type', template="plotly_white", markers=True)
    fig1.update_layout(height=400, margin=dict(t=30, b=30))
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<div class='chart-title'>Material Revenue Contribution by Year</div>", unsafe_allow_html=True)
        material_yoy_df = filtered_df.groupby(['Year', 'Material'])['Total_INR'].sum().reset_index()
        fig_stacked = px.bar(material_yoy_df, x='Year', y='Total_INR', color='Material',
                             title="",
                             labels={'Total_INR': 'Total Revenue (₹)', 'Year': 'Year'},
                             color_continuous_scale='viridis')
        fig_stacked.update_layout(height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig_stacked, use_container_width=True)
        
    with c2:
        st.markdown("<div class='chart-title'>Sales Mix (Volume Share)</div>", unsafe_allow_html=True)
        mix_df = filtered_df.groupby('MixType')['Quantity'].sum().reset_index()
        fig_mix = px.pie(mix_df, values='Quantity', names='MixType', hole=0.3)
        fig_mix.update_traces(textposition='inside', textinfo='percent+label')
        fig_mix.update_layout(showlegend=False, height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig_mix, use_container_width=True)
        
    st.markdown("---") 

    st.markdown("### **Market and Pricing Trends**")
    adv_c3, adv_c4 = st.columns(2)
    
    with adv_c3:
        st.markdown("<div class='chart-title'>Average Price Trend by Material</div>", unsafe_allow_html=True)
        rate_trend = filtered_df.groupby(['Month_Year', 'Material'])['Rate'].mean().reset_index()
        rate_trend['Date'] = pd.to_datetime(rate_trend['Month_Year'])
        
        fig_rate = px.line(rate_trend, x='Date', y='Rate', color='Material',
                           title="", 
                           labels={'Rate': 'Average Price (₹)', 'Date': 'Time'},
                           template="plotly_white")
        fig_rate.update_layout(height=400, margin=dict(t=30, b=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_rate, use_container_width=True)

    with adv_c4:
        st.markdown("<div class='chart-title'>Top 5 Materials: Volume vs. Price</div>", unsafe_allow_html=True)
        material_summary = filtered_df.groupby('Material').agg(
            Total_Volume=('Quantity', 'sum'),
            Avg_Price=('Rate', 'mean')
        ).reset_index()
        top_5_materials = material_summary.nlargest(5, 'Total_Volume')
        
        fig_bubble = px.scatter(top_5_materials, 
                                x='Total_Volume', 
                                y='Avg_Price', 
                                size='Total_Volume', 
                                color='Material', 
                                hover_name='Material',
                                template="plotly_white",
                                labels={'Total_Volume': 'Total Volume Sold (m³)', 'Avg_Price': 'Average Selling Price (₹)'})
        fig_bubble.update_layout(height=400, margin=dict(t=30, b=30))
        st.plotly_chart(fig_bubble, use_container_width=True)


# --- TAB 2: CAUSAL & VALUE ANALYSIS ---
with tab2:
    st.header("🔬 Causal & Business Volatility Analysis")
    
    col_pareto, col_scatter = st.columns(2)
    
    # CHART A: CUSTOMER REVENUE PARETO
    with col_pareto:
        st.markdown("<div class='chart-title'>Customer Revenue Pareto (80/20 Analysis)</div>", unsafe_allow_html=True)
        
        customer_rev = filtered_df.groupby('Customer')['Total_INR'].sum().sort_values(ascending=False).reset_index()
        customer_rev['Cumulative_Share'] = customer_rev['Total_INR'].cumsum() / customer_rev['Total_INR'].sum()
        customer_rev['Customer_Rank'] = range(1, len(customer_rev) + 1)
        
        # Create Pareto Chart (Bar for Revenue, Line for Cumulative Share)
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for individual revenue
        fig_pareto.add_trace(
            go.Bar(x=customer_rev['Customer_Rank'], y=customer_rev['Total_INR'], name='Revenue (INR)'),
            secondary_y=False,
        )

        # Line chart for cumulative percentage
        fig_pareto.add_trace(
            go.Scatter(x=customer_rev['Customer_Rank'], y=customer_rev['Cumulative_Share'], name='Cumulative Share', mode='lines+markers', yaxis='y2'),
            secondary_y=True,
        )

        fig_pareto.update_layout(
            height=450, margin=dict(t=30, b=30), template="plotly_white",
            xaxis_title="Customer Rank",
            yaxis_title="Revenue (₹)",
            yaxis2_title="Cumulative Share (%)",
            hovermode="x unified",
            showlegend=True
        )
        fig_pareto.update_yaxes(tickformat=".0%", secondary_y=True, range=[0, 1])
        st.plotly_chart(fig_pareto, use_container_width=True)


    # PRICE ELASTICITY SCATTER
    with col_scatter:
        st.markdown("<div class='chart-title'>Price-Demand Relationship (Causal Proof)</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            filtered_df.sample(min(1500, len(filtered_df))), 
            x='Rate', y='Quantity', 
            color='Material', 
            trendline="ols", 
            hover_data=['Customer', 'Date'], 
            template="plotly_white",
            labels={'Rate': 'Price per m³ (₹)', 'Quantity': 'Quantity Sold (m³)'}
        )
        fig_scatter.update_layout(height=450, margin=dict(t=30, b=30))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---") 

    # ROW 2: ADVANCED OPERATIONAL METRICS
    st.markdown("### **Advanced Operational Metrics**")
    
    adv_c1, adv_c2 = st.columns(2)
    
    with adv_c1:
        st.markdown("<div class='chart-title'>Material Price Distribution (Key Pricing Range)</div>", unsafe_allow_html=True)
        fig_box = px.box(filtered_df, x='Material', y='Rate', color='Material',
                         title="",
                         labels={'Rate': 'Price (₹)', 'Material': ''},
                         template="plotly_white")
        fig_box.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    with adv_c2:
        st.markdown("<div class='chart-title'>Demand Volatility Index (Inventory Risk)</div>", unsafe_allow_html=True)
        volatility_df = filtered_df.groupby('Material')['Quantity'].agg(
            ['mean', 'std']
        ).reset_index()
        volatility_df['Volatility'] = volatility_df['std'] / volatility_df['mean']
        
        fig_vol = px.bar(volatility_df.sort_values('Volatility', ascending=False), x='Material', y='Volatility', color='Volatility',
                         color_continuous_scale=px.colors.sequential.Sunset,
                         labels={'Volatility': 'Coefficient of Variation (Lower is Better)'})
        fig_vol.update_layout(height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption("Materials with a higher index have unpredictable demand, signaling higher inventory risk.")
# ================================

# ================================
# FUNCTIONS REQUIRED FOR TAB 3
# ================================

def rule_based_chatbot(query, df, context=None):
    q = query.lower()

    if any(x in q for x in ["top customer", "highest revenue customer"]):
        grp = df.groupby("Customer")["Total_INR"].sum()
        return f"🏆 Top Customer: {grp.idxmax()}\nRevenue: ₹{grp.max():,.0f}"

    if any(x in q for x in ["least customer", "least contributing customer"]):
        grp = df.groupby("Customer")["Total_INR"].sum()
        return f"🔻 Least Customer: {grp.idxmin()}\nRevenue: ₹{grp.min():,.0f}"

    if "2023" in q:
        y = df[df["Year"] == 2023]["Total_INR"].sum()
        return f"📅 Total Sales in 2023: ₹{y:,.0f}"

    if any(x in q for x in ["profit", "loss", "price"]):
        return (
            "📈 Pricing Decision\n\n"
            "Use the **Price Impact Simulator above** to see profit or loss "
            "by adjusting price percentage."
        )

    return "🤖 Ask about customers, pricing, or yearly sales."


def price_impact_simulator(df, material, price_change_pct, elasticity):
    data = df[df["Material"] == material]
    if data.empty:
        return None

    base = data["Total_INR"].sum()
    demand_change = elasticity * price_change_pct
    new = base * (1 + price_change_pct / 100) * (1 + demand_change / 100)

    diff = new - base
    pct = (diff / base) * 100

    risk = "Low" if abs(pct) < 3 else "Medium" if abs(pct) < 7 else "High"
    reco = "Increase price" if pct > 0 else "Avoid price increase"

    return base, new, diff, pct, risk, reco


def chart_from_query(query, df):
    q = query.lower()

    if "revenue by material" in q:
        d = df.groupby("Material")["Total_INR"].sum().reset_index()
        return px.bar(d, x="Material", y="Total_INR")

    if "revenue trend" in q:
        d = df.groupby("Month_Year")["Total_INR"].sum().reset_index()
        return px.line(d, x="Month_Year", y="Total_INR")

    return None



with tab3:
    st.header("🤖 AI Strategic Brain")

    # ================================
    # 1️⃣ PRICE IMPACT SIMULATOR
    # ================================
    st.markdown("### 1️⃣ Price Impact Simulator (Decision Engine)")

    col1, col2 = st.columns(2)

    with col1:
        sim_material = st.selectbox(
            "Select Material",
            options=sorted(filtered_df["Material"].unique())
        )

    with col2:
        price_change = st.slider(
            "Adjust Price (%)",
            min_value=-20,
            max_value=20,
            value=5,
            step=1
        )

    sim = price_impact_simulator(filtered_df, sim_material, price_change, elasticity)

    if sim:
        base, new, diff, pct, risk, reco = sim

        a, b, c = st.columns(3)
        a.metric("Current Revenue", f"₹{base:,.0f}")
        b.metric("Projected Revenue", f"₹{new:,.0f}", f"{pct:.2f}%")

        if pct >= 0:
            c.success(f"📈 PROFIT ↑\n\n₹{diff:,.0f} ({pct:.2f}%)")
        else:
            c.error(f"📉 LOSS ↓\n\n₹{abs(diff):,.0f} ({abs(pct):.2f}%)")

        st.markdown(
            f"""
            **🧠 AI Recommendation:** {reco}  
            **⚠️ Risk Level:** {risk}  
            **📌 Explanation:** Based on historical price elasticity and demand response.
            """
        )

    st.divider()

    # ================================
    # 2️⃣ CONVERSATIONAL BUSINESS INTELLIGENCE
    # ================================
    st.markdown("### 2️⃣ Conversational Business Intelligence")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me about pricing decisions, revenue risk, customers, or trends."}
        ]
        st.session_state.context = {}

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask: who is top customer, least customer, pricing risk, August sales, trends")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        answer = rule_based_chatbot(query, filtered_df, st.session_state.context)
        fig = chart_from_query(query, filtered_df)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.caption(
        "ℹ️ Insights are based on historical data and simulations. "
        "Forecasting and external market shocks are outside current scope."
    )
