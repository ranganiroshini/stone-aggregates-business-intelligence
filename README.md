# Stone Aggregates Business Intelligence Dashboard
🔗 **Live Application:**  
https://stone-aggregates-business-intelligence-cwnjxyyesmqz98nrcepv39.streamlit.app/

This dashboard was developed using **actual business data from a stone aggregates company**, with the goal of:
- Improving visibility into revenue, pricing, and demand
- Supporting smarter pricing and operational decisions
- Replacing manual Excel-based analysis with an interactive analytics platform
- Demonstrating how real businesses can adopt data analytics effectively
  ## What This Dashboard Solves
- Revenue tracking across materials, customers, and time
- Identification of top customers and revenue concentration risk
- Understanding price–demand relationships (elasticity)
- Forecasting short-term revenue trends
- Simulating pricing decisions and their impact
- Providing instant answers via a conversational BI chatbot
  ##  Key Features
### Executive KPIs
- Total Revenue
- Average Load Price
- Revenue Concentration (Top 5 Customers)
- Average Units per Load

### Predictive & Trend Analysis
- Monthly revenue trends
- 3-month forward revenue forecasting
- Material-wise revenue contribution
- Sales mix analysis

### Causal & Risk Analysis
- Price vs Demand relationship (Elasticity)
- Customer Pareto (80/20 revenue analysis)
- Demand volatility index (inventory risk indicator)

### AI-Powered Decision Intelligence
- **Price Impact Simulator** using historical price elasticity
- Risk classification (Low / Medium / High)
- **Conversational BI chatbot** for quick business queries such as:
  - Top customer
  - Least contributing customer
  - Yearly revenue
  - Pricing insights
 
  ## Data Source & Transparency

### Raw Data Origin
- The project starts with **raw operational transaction data**
- Data includes:
  - Date
  - Material type
  - Customer
  - Quantity (m³)
  - Price / Rate
  - Total revenue (INR)
  ### Initial Data Handling (Excel)
- Raw data was first reviewed and structured in **Microsoft Excel**
- Column validation, formatting, and consistency checks were performed
- This ensured clean ingestion into Python
## Data Processing & Analytics Pipeline
Raw Business Data
↓
Excel (Initial Cleaning & Validation)
↓
Python (Anaconda Environment)
↓
Data Cleaning & Feature Engineering (Pandas)
↓
KPI & Aggregation Logic
↓
Predictive & Causal Analytics
↓
Streamlit Dashboard Deployment
##  Technologies Used
| Category | Tools |
|-------|------
| Programming Language | Python |
| Environment | Anaconda |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Dashboard Framework | Streamlit |
| Forecasting Logic | Statistical averaging |
| Causal Analysis | Price Elasticity |
| UI Styling | Custom CSS |
| Deployment | Streamlit Cloud |


##  AI & Business Logic Explained

###  Price Elasticity Model
- Uses historical price vs quantity data
- Estimates demand response to price changes
- Forms the foundation of the pricing simulator

###  Price Impact Simulator
- User selects material and price change %
- System calculates:
  - New projected revenue
  - Revenue gain or loss
  - Risk level (Low / Medium / High)
- Designed for **real pricing decision scenarios**

### Conversational BI Chatbot
- Rule-based AI assistant
- Answers common business questions instantly
- Designed to reduce dependency on manual analysis
---
## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
