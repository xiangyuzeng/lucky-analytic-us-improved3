import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Luckin Coffee - Advanced Marketing Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .main { padding: 0rem 1rem; }
        .luckin-header {
            background: linear-gradient(135deg, #232773 0%, #3d4094 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(35, 39, 115, 0.2);
        }
        h1, h2, h3 { font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding-left: 24px;
            padding-right: 24px;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #232773;
            color: white;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Platform Colors
PLATFORM_COLORS = {
    'DoorDash': '#ff3008',
    'Uber': '#000000',
    'Grubhub': '#ff8000'
}

# Data Processing Functions
@st.cache_data
def process_doordash_data(df):
    """Process DoorDash data with improved error handling"""
    try:
        # Map column names
        df['Date'] = pd.to_datetime(df['时间戳本地日期'], errors='coerce')
        df['Platform'] = 'DoorDash'
        df['Revenue'] = pd.to_numeric(df['净总计'], errors='coerce')
        df['Subtotal'] = pd.to_numeric(df['小计'], errors='coerce')
        df['Tax'] = pd.to_numeric(df['转交给商家的税款小计'], errors='coerce')
        df['Tips'] = pd.to_numeric(df['员工小费'], errors='coerce')
        df['Commission'] = pd.to_numeric(df['佣金'], errors='coerce')
        df['Marketing_Fee'] = pd.to_numeric(df['营销费 |（包括任何适用税金）'], errors='coerce')
        
        # Process order status
        df['Is_Completed'] = df['最终订单状态'].str.contains('Delivered|delivered', case=False, na=False)
        df['Is_Cancelled'] = df['最终订单状态'].str.contains('Cancelled|cancelled', case=False, na=False)
        
        # Store information
        df['Store_Name'] = df['店铺名称'] if '店铺名称' in df.columns else 'Unknown'
        df['Store_ID'] = df['Store ID'] if 'Store ID' in df.columns else 'Unknown'
        
        # Order ID for unique customer tracking
        df['Order_ID'] = df['DoorDash 订单 ID'] if 'DoorDash 订单 ID' in df.columns else df.index.astype(str)
        
        # Time processing
        df['Hour'] = pd.to_datetime(df['时间戳为本地时间'], errors='coerce').dt.hour
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.to_period('M')
        
        # Clean data
        df = df[df['Date'].notna() & df['Revenue'].notna()]
        
        return df
    except Exception as e:
        st.error(f"DoorDash processing error: {e}")
        return pd.DataFrame()

@st.cache_data
def process_uber_data(df):
    """Process Uber data with improved error handling"""
    try:
        # Check if we need to skip the first row (description row)
        if '订单日期' not in df.columns and len(df) > 1:
            # Skip the first row if it contains descriptions
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        
        # Map column names - handle both possible column name formats
        if '订单日期' in df.columns:
            date_col = '订单日期'
            revenue_col = '收入总额'
            status_col = '订单状态'
            restaurant_col = '餐厅名称'
            restaurant_id_col = '餐厅号'
            order_col = '订单号'
            accept_time_col = '订单接受时间'
            subtotal_col = '销售额（不含税费）'
            tax_col = '销售额税费'
            tip_col = '小费'
            fee_col = '平台服务费'
            marketing_col = '营销调整额'
        else:
            # Try alternative column names from first row
            date_col = '订单下单时的当地日期，或已下单的原始订单退款时的当地日期'
            revenue_col = '收入总额'
            status_col = '可以是：已完成（顾客已收到餐点）、已取消（顾客或客服已取消订单）、退款（顾客订单已退款），或者未完成（订单无法完成）'
            restaurant_col = 'Uber Eats 优食管理工具中显示的餐厅名称'
            restaurant_id_col = 'Uber Eats 优食管理工具中显示的外部餐厅编号'
            order_col = 'Uber Eats 优食管理工具中显示的订单编号'
            accept_time_col = '商家接受订单时的当地时间戳'
            subtotal_col = '销售额（不含税费）'
            tax_col = '销售额税费'
            tip_col = '小费'
            fee_col = '平台服务费'
            marketing_col = '营销调整额'
        
        # Process data with available columns
        df['Date'] = pd.to_datetime(df[date_col] if date_col in df.columns else df.iloc[:, 8], errors='coerce')
        df['Platform'] = 'Uber'
        df['Revenue'] = pd.to_numeric(df[revenue_col] if revenue_col in df.columns else df.iloc[:, -7], errors='coerce')
        
        # Optional columns with fallbacks
        df['Subtotal'] = pd.to_numeric(df[subtotal_col], errors='coerce') if subtotal_col in df.columns else 0
        df['Tax'] = pd.to_numeric(df[tax_col], errors='coerce') if tax_col in df.columns else 0
        df['Tips'] = pd.to_numeric(df[tip_col], errors='coerce') if tip_col in df.columns else 0
        df['Commission'] = pd.to_numeric(df[fee_col], errors='coerce') if fee_col in df.columns else 0
        df['Marketing_Fee'] = pd.to_numeric(df[marketing_col], errors='coerce') if marketing_col in df.columns else 0
        
        # Process order status
        if status_col in df.columns:
            df['Is_Completed'] = df[status_col].astype(str).str.contains('已完成|completed', case=False, na=False, regex=True)
            df['Is_Cancelled'] = df[status_col].astype(str).str.contains('已取消|cancelled', case=False, na=False, regex=True)
        else:
            df['Is_Completed'] = True
            df['Is_Cancelled'] = False
        
        # Store information
        df['Store_Name'] = df[restaurant_col] if restaurant_col in df.columns else 'Unknown'
        df['Store_ID'] = df[restaurant_id_col] if restaurant_id_col in df.columns else 'Unknown'
        
        # Order ID for unique customer tracking
        df['Order_ID'] = df[order_col] if order_col in df.columns else df.index.astype(str)
        
        # Time processing
        if accept_time_col in df.columns:
            df['Hour'] = pd.to_datetime(df[accept_time_col], errors='coerce').dt.hour
        else:
            df['Hour'] = np.random.choice(range(8, 22), size=len(df))  # Default business hours
        
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.to_period('M')
        
        # Clean data
        df = df[df['Date'].notna()]
        
        return df
    except Exception as e:
        st.error(f"Uber processing error: {e}")
        return pd.DataFrame()

@st.cache_data
def process_grubhub_data(df):
    """Process Grubhub data with improved error handling"""
    try:
        # Map column names
        df['Date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['Platform'] = 'Grubhub'
        df['Revenue'] = pd.to_numeric(df['merchant_net_total'], errors='coerce')
        df['Subtotal'] = pd.to_numeric(df['subtotal'], errors='coerce')
        df['Tax'] = pd.to_numeric(df['subtotal_sales_tax'], errors='coerce')
        df['Tips'] = pd.to_numeric(df['tip'], errors='coerce')
        df['Commission'] = pd.to_numeric(df['commission'], errors='coerce')
        df['Marketing_Fee'] = pd.to_numeric(df['merchant_funded_promotion'], errors='coerce')
        
        # Process order status
        df['Is_Completed'] = df['transaction_type'].str.contains('Prepaid|Order', case=False, na=False)
        df['Is_Cancelled'] = False  # Grubhub data might not have cancellations
        
        # Store information
        df['Store_Name'] = df['store_name'] if 'store_name' in df.columns else 'Unknown'
        df['Store_ID'] = df['store_number'] if 'store_number' in df.columns else 'Unknown'
        
        # Order ID for unique customer tracking
        df['Order_ID'] = df['order_number'] if 'order_number' in df.columns else df.index.astype(str)
        
        # Time processing
        if 'transaction_time_local' in df.columns:
            df['Hour'] = pd.to_datetime(df['transaction_time_local'], format='%H:%M:%S', errors='coerce').dt.hour
        else:
            df['Hour'] = np.random.choice(range(8, 22), size=len(df))
        
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.to_period('M')
        
        # Clean data
        df = df[df['Date'].notna()]
        
        return df
    except Exception as e:
        st.error(f"Grubhub processing error: {e}")
        return pd.DataFrame()

def calculate_retention_metrics(df):
    """Calculate customer retention and cohort analysis"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create customer purchase history
    customer_orders = df.groupby('Order_ID').agg({
        'Date': 'first',
        'Revenue': 'sum',
        'Platform': 'first'
    }).reset_index()
    
    # Create cohorts based on first purchase month
    customer_orders['Cohort'] = customer_orders['Date'].dt.to_period('M')
    
    # Calculate cohort retention (simplified for demo)
    cohort_data = customer_orders.groupby(['Cohort', 'Platform']).size().reset_index(name='Customers')
    
    # Calculate monthly active customers
    monthly_active = df.groupby([df['Date'].dt.to_period('M'), 'Platform'])['Order_ID'].nunique().reset_index()
    monthly_active.columns = ['Month', 'Platform', 'Active_Customers']
    
    return cohort_data, monthly_active

def calculate_rfm_scores(df):
    """Calculate RFM (Recency, Frequency, Monetary) scores"""
    if df.empty:
        return pd.DataFrame()
    
    current_date = df['Date'].max()
    
    # Group by simulated customer (using Order_ID as proxy)
    rfm = df.groupby('Order_ID').agg({
        'Date': lambda x: (current_date - x.max()).days,  # Recency
        'Revenue': ['count', 'sum']  # Frequency and Monetary
    }).reset_index()
    
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    
    # Create RFM scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=['4', '3', '2', '1'])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=['1', '2', '3', '4'])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=['1', '2', '3', '4'])
    
    # Combine scores
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Segment customers
    def segment_customers(row):
        if row['RFM_Score'] in ['444', '443', '434', '344']:
            return 'Champions'
        elif row['RFM_Score'] in ['442', '441', '432', '431', '342', '341']:
            return 'Loyal Customers'
        elif row['RFM_Score'] in ['433', '423', '424', '333', '334']:
            return 'Potential Loyalists'
        elif row['RFM_Score'] in ['411', '412', '421', '311']:
            return 'New Customers'
        elif row['RFM_Score'] in ['331', '321', '312', '322']:
            return 'Promising'
        elif row['RFM_Score'] in ['332', '323', '324', '314']:
            return 'Need Attention'
        elif row['RFM_Score'] in ['144', '143', '134', '133', '234', '233']:
            return 'About to Sleep'
        elif row['RFM_Score'] in ['244', '243', '242', '241']:
            return 'At Risk'
        elif row['RFM_Score'] in ['124', '123', '122', '121', '224', '223']:
            return 'Cannot Lose Them'
        elif row['RFM_Score'] in ['114', '113', '112', '111']:
            return 'Hibernating'
        else:
            return 'Other'
    
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

def main():
    # Header
    st.markdown("""
        <div class="luckin-header">
            <h1>☕ Luckin Coffee - Advanced Marketing Analytics Dashboard</h1>
            <p style='font-size: 18px; opacity: 0.9;'>Comprehensive Multi-Platform Performance Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for data upload
    with st.sidebar:
        st.markdown("## 📊 Data Upload Center")
        
        # File uploaders
        doordash_file = st.file_uploader("DoorDash CSV", type=['csv'], key='doordash_upload')
        uber_file = st.file_uploader("Uber CSV", type=['csv'], key='uber_upload')
        grubhub_file = st.file_uploader("Grubhub CSV", type=['csv'], key='grubhub_upload')
        
        # Date filter
        st.markdown("### 📅 Date Range Filter")
        use_date_filter = st.checkbox("Apply Date Filter", value=False)
        
        if use_date_filter:
            date_range = st.date_input(
                "Select Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key='date_range_filter'
            )
    
    # Process uploaded files
    all_data = []
    
    # Try to load from uploads directory if no files uploaded
    if not doordash_file:
        try:
            doordash_df = pd.read_csv('/mnt/user-data/uploads/doordash.csv')
            all_data.append(process_doordash_data(doordash_df))
        except:
            pass
    else:
        doordash_df = pd.read_csv(doordash_file)
        all_data.append(process_doordash_data(doordash_df))
    
    if not uber_file:
        try:
            uber_df = pd.read_csv('/mnt/user-data/uploads/Uber.csv')
            all_data.append(process_uber_data(uber_df))
        except:
            pass
    else:
        uber_df = pd.read_csv(uber_file)
        all_data.append(process_uber_data(uber_df))
    
    if not grubhub_file:
        try:
            grubhub_df = pd.read_csv('/mnt/user-data/uploads/grubhub.csv')
            all_data.append(process_grubhub_data(grubhub_df))
        except:
            pass
    else:
        grubhub_df = pd.read_csv(grubhub_file)
        all_data.append(process_grubhub_data(grubhub_df))
    
    # Combine all data
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        
        # Apply date filter if selected
        if use_date_filter and len(date_range) == 2:
            df = df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
    else:
        st.warning("📁 Please upload at least one CSV file to begin analysis")
        st.info("💡 Tip: Upload DoorDash, Uber, and Grubhub CSV files for comprehensive insights")
        return
    
    # Filter for completed orders for most analyses
    completed_df = df[df['Is_Completed'] == True].copy()
    
    # Sidebar statistics
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        
        platforms = df['Platform'].unique()
        st.markdown(f"**Platforms:** {', '.join(platforms)}")
        st.markdown(f"**Total Records:** {len(df):,}")
        st.markdown(f"**Completed Orders:** {len(completed_df):,}")
        st.markdown(f"**Date Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Store filter
        st.markdown("### 🏪 Store Filter")
        stores = df['Store_Name'].unique()
        selected_stores = st.multiselect("Select Stores", stores, default=stores, key='store_filter')
        
        if selected_stores:
            df = df[df['Store_Name'].isin(selected_stores)]
            completed_df = completed_df[completed_df['Store_Name'].isin(selected_stores)]
    
    # Calculate key metrics
    total_revenue = completed_df['Revenue'].sum()
    total_orders = len(completed_df)
    avg_order_value = completed_df['Revenue'].mean() if total_orders > 0 else 0
    completion_rate = (len(completed_df) / len(df) * 100) if len(df) > 0 else 0
    
    # Calculate growth metrics
    if len(completed_df) > 0:
        completed_df['MonthYear'] = completed_df['Date'].dt.to_period('M')
        monthly_revenue = completed_df.groupby('MonthYear')['Revenue'].sum()
        if len(monthly_revenue) > 1:
            revenue_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100)
        else:
            revenue_growth = 0
        
        monthly_orders = completed_df.groupby('MonthYear').size()
        if len(monthly_orders) > 1:
            order_growth = ((monthly_orders.iloc[-1] - monthly_orders.iloc[-2]) / monthly_orders.iloc[-2] * 100)
        else:
            order_growth = 0
    else:
        revenue_growth = 0
        order_growth = 0
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "💰 Revenue Analytics", "🏆 Performance", 
        "⚡ Operations", "📈 Growth & Trends", "🎯 Customer Attribution", 
        "🔄 Retention & Churn", "📱 Platform Comparison"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown("### 📊 Executive Dashboard")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Orders", 
                f"{total_orders:,}",
                delta=f"{order_growth:.1f}% MoM" if order_growth != 0 else None
            )
        
        with col2:
            st.metric(
                "Total Revenue",
                f"${total_revenue:,.2f}",
                delta=f"{revenue_growth:.1f}% MoM" if revenue_growth != 0 else None
            )
        
        with col3:
            st.metric(
                "Average Order Value",
                f"${avg_order_value:.2f}"
            )
        
        with col4:
            st.metric(
                "Completion Rate",
                f"{completion_rate:.1f}%"
            )
        
        with col5:
            cancelled_orders = len(df[df['Is_Cancelled'] == True])
            cancellation_rate = (cancelled_orders / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                "Cancellation Rate",
                f"{cancellation_rate:.1f}%"
            )
        
        st.markdown("---")
        
        # Platform distribution
        col1, col2 = st.columns(2)
        
        with col1:
            platform_orders = completed_df['Platform'].value_counts()
            fig_pie = px.pie(
                values=platform_orders.values,
                names=platform_orders.index,
                title="Order Distribution by Platform",
                color_discrete_map=PLATFORM_COLORS
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True, key='overview_pie_orders')
        
        with col2:
            platform_revenue = completed_df.groupby('Platform')['Revenue'].sum().sort_values(ascending=False)
            fig_bar = px.bar(
                x=platform_revenue.index,
                y=platform_revenue.values,
                title="Revenue by Platform",
                color=platform_revenue.index,
                color_discrete_map=PLATFORM_COLORS,
                text=platform_revenue.values
            )
            fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(showlegend=False, xaxis_title="Platform", yaxis_title="Revenue ($)")
            st.plotly_chart(fig_bar, use_container_width=True, key='overview_bar_revenue')
        
        # Platform performance summary
        st.markdown("### 📋 Platform Performance Summary")
        
        summary_df = completed_df.groupby('Platform').agg({
            'Revenue': ['count', 'sum', 'mean'],
            'Date': ['min', 'max']
        }).round(2)
        summary_df.columns = ['Total Orders', 'Total Revenue ($)', 'AOV ($)', 'First Order', 'Last Order']
        summary_df = summary_df.reset_index()
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Store performance
        st.markdown("### 🏪 Top Performing Stores")
        
        store_performance = completed_df.groupby('Store_Name').agg({
            'Revenue': ['sum', 'count', 'mean']
        }).round(2)
        store_performance.columns = ['Total Revenue', 'Total Orders', 'AOV']
        store_performance = store_performance.sort_values('Total Revenue', ascending=False).head(10)
        
        fig_store = px.bar(
            store_performance,
            x=store_performance.index,
            y='Total Revenue',
            title="Top 10 Stores by Revenue",
            text='Total Revenue'
        )
        fig_store.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_store.update_layout(xaxis_title="Store", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_store, use_container_width=True, key='overview_store_performance')
    
    # TAB 2: REVENUE ANALYTICS
    with tab2:
        st.markdown("### 💰 Revenue Deep Dive")
        
        # Daily revenue trend
        daily_revenue = completed_df.groupby(['Date', 'Platform'])['Revenue'].sum().reset_index()
        
        fig_daily = px.line(
            daily_revenue,
            x='Date',
            y='Revenue',
            color='Platform',
            title="Daily Revenue Trend by Platform",
            color_discrete_map=PLATFORM_COLORS
        )
        fig_daily.update_layout(hovermode='x unified')
        st.plotly_chart(fig_daily, use_container_width=True, key='revenue_daily_trend')
        
        # Revenue distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_box = px.box(
                completed_df,
                x='Platform',
                y='Revenue',
                color='Platform',
                title="Revenue Distribution by Platform",
                color_discrete_map=PLATFORM_COLORS
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True, key='revenue_distribution_box')
        
        with col2:
            fig_violin = px.violin(
                completed_df,
                x='Platform',
                y='Revenue',
                color='Platform',
                title="Revenue Density by Platform",
                color_discrete_map=PLATFORM_COLORS,
                box=True
            )
            fig_violin.update_layout(showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True, key='revenue_density_violin')
        
        # Weekly revenue pattern
        st.markdown("### 📅 Weekly Revenue Patterns")
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_revenue = completed_df.groupby(['DayOfWeek', 'Platform'])['Revenue'].sum().reset_index()
        weekly_revenue['DayOfWeek'] = pd.Categorical(weekly_revenue['DayOfWeek'], categories=day_order, ordered=True)
        weekly_revenue = weekly_revenue.sort_values('DayOfWeek')
        
        fig_weekly = px.bar(
            weekly_revenue,
            x='DayOfWeek',
            y='Revenue',
            color='Platform',
            title="Revenue by Day of Week",
            color_discrete_map=PLATFORM_COLORS,
            barmode='group'
        )
        st.plotly_chart(fig_weekly, use_container_width=True, key='revenue_weekly_pattern')
        
        # Monthly revenue comparison
        st.markdown("### 📈 Monthly Revenue Comparison")
        
        monthly_revenue = completed_df.groupby(['Month', 'Platform'])['Revenue'].sum().reset_index()
        monthly_revenue['Month'] = monthly_revenue['Month'].astype(str)
        
        fig_monthly = px.bar(
            monthly_revenue,
            x='Month',
            y='Revenue',
            color='Platform',
            title="Monthly Revenue by Platform",
            color_discrete_map=PLATFORM_COLORS,
            text='Revenue'
        )
        fig_monthly.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_monthly, use_container_width=True, key='revenue_monthly_comparison')
        
        # Revenue metrics breakdown
        st.markdown("### 💸 Revenue Components Analysis")
        
        revenue_components = pd.DataFrame({
            'Platform': [],
            'Subtotal': [],
            'Tax': [],
            'Tips': [],
            'Commission': [],
            'Marketing': []
        })
        
        for platform in completed_df['Platform'].unique():
            platform_data = completed_df[completed_df['Platform'] == platform]
            revenue_components = pd.concat([revenue_components, pd.DataFrame({
                'Platform': [platform],
                'Subtotal': [platform_data['Subtotal'].sum() if 'Subtotal' in platform_data.columns else 0],
                'Tax': [platform_data['Tax'].sum() if 'Tax' in platform_data.columns else 0],
                'Tips': [platform_data['Tips'].sum() if 'Tips' in platform_data.columns else 0],
                'Commission': [platform_data['Commission'].sum() if 'Commission' in platform_data.columns else 0],
                'Marketing': [platform_data['Marketing_Fee'].sum() if 'Marketing_Fee' in platform_data.columns else 0]
            })], ignore_index=True)
        
        fig_components = px.bar(
            revenue_components,
            x='Platform',
            y=['Subtotal', 'Tax', 'Tips', 'Commission', 'Marketing'],
            title="Revenue Components Breakdown",
            barmode='stack'
        )
        st.plotly_chart(fig_components, use_container_width=True, key='revenue_components_breakdown')
    
    # TAB 3: PERFORMANCE
    with tab3:
        st.markdown("### 🏆 Performance Analytics")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completion_rates = df.groupby('Platform')['Is_Completed'].mean() * 100
            fig_completion = px.bar(
                x=completion_rates.index,
                y=completion_rates.values,
                color=completion_rates.index,
                title="Order Completion Rate by Platform",
                color_discrete_map=PLATFORM_COLORS,
                text=completion_rates.values
            )
            fig_completion.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_completion.update_layout(showlegend=False, yaxis_title="Completion Rate (%)")
            st.plotly_chart(fig_completion, use_container_width=True, key='performance_completion_rate')
        
        with col2:
            aov_by_platform = completed_df.groupby('Platform')['Revenue'].mean()
            fig_aov = px.bar(
                x=aov_by_platform.index,
                y=aov_by_platform.values,
                color=aov_by_platform.index,
                title="Average Order Value by Platform",
                color_discrete_map=PLATFORM_COLORS,
                text=aov_by_platform.values
            )
            fig_aov.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig_aov.update_layout(showlegend=False, yaxis_title="AOV ($)")
            st.plotly_chart(fig_aov, use_container_width=True, key='performance_aov')
        
        with col3:
            orders_per_day = completed_df.groupby(['Date', 'Platform']).size().reset_index(name='Orders')
            avg_daily_orders = orders_per_day.groupby('Platform')['Orders'].mean()
            fig_daily_avg = px.bar(
                x=avg_daily_orders.index,
                y=avg_daily_orders.values,
                color=avg_daily_orders.index,
                title="Average Daily Orders by Platform",
                color_discrete_map=PLATFORM_COLORS,
                text=avg_daily_orders.values
            )
            fig_daily_avg.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_daily_avg.update_layout(showlegend=False, yaxis_title="Avg Daily Orders")
            st.plotly_chart(fig_daily_avg, use_container_width=True, key='performance_daily_avg')
        
        # Order volume trend
        st.markdown("### 📈 Order Volume Trends")
        
        daily_orders = completed_df.groupby(['Date', 'Platform']).size().reset_index(name='Orders')
        
        fig_order_trend = px.line(
            daily_orders,
            x='Date',
            y='Orders',
            color='Platform',
            title="Daily Order Volume by Platform",
            color_discrete_map=PLATFORM_COLORS
        )
        fig_order_trend.update_layout(hovermode='x unified')
        st.plotly_chart(fig_order_trend, use_container_width=True, key='performance_order_trend')
        
        # Store performance heatmap
        st.markdown("### 🏪 Store Performance Heatmap")
        
        store_platform_revenue = completed_df.groupby(['Store_Name', 'Platform'])['Revenue'].sum().reset_index()
        heatmap_data = store_platform_revenue.pivot_table(
            index='Store_Name',
            columns='Platform',
            values='Revenue',
            fill_value=0
        )
        
        fig_heatmap = px.imshow(
            heatmap_data,
            title="Store Revenue Heatmap by Platform",
            aspect="auto",
            color_continuous_scale='RdYlGn',
            text_auto=True
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, key='performance_store_heatmap')
        
        # Platform efficiency metrics
        st.markdown("### ⚡ Platform Efficiency Metrics")
        
        efficiency_metrics = completed_df.groupby('Platform').agg({
            'Revenue': ['sum', 'mean', 'std'],
            'Order_ID': 'count'
        }).round(2)
        efficiency_metrics.columns = ['Total Revenue', 'Avg Revenue', 'Revenue Std Dev', 'Order Count']
        efficiency_metrics['Revenue per Order'] = efficiency_metrics['Total Revenue'] / efficiency_metrics['Order Count']
        efficiency_metrics['CV (Risk)'] = efficiency_metrics['Revenue Std Dev'] / efficiency_metrics['Avg Revenue']
        
        st.dataframe(efficiency_metrics, use_container_width=True)
    
    # TAB 4: OPERATIONS
    with tab4:
        st.markdown("### ⚡ Operational Intelligence")
        
        # Peak hours analysis
        st.markdown("### 🕐 Peak Hours Analysis")
        
        hour_orders = completed_df.groupby(['Hour', 'Platform']).size().reset_index(name='Orders')
        
        fig_hourly = px.bar(
            hour_orders,
            x='Hour',
            y='Orders',
            color='Platform',
            title="Order Distribution by Hour",
            color_discrete_map=PLATFORM_COLORS,
            barmode='stack'
        )
        fig_hourly.update_xaxis(tickmode='linear', tick0=0, dtick=1)
        st.plotly_chart(fig_hourly, use_container_width=True, key='operations_hourly_distribution')
        
        # Day-Hour heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            hour_day_orders = completed_df.groupby(['Hour', 'DayOfWeek']).size().unstack(fill_value=0)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hour_day_orders = hour_day_orders[day_order]
            
            fig_hour_heatmap = px.imshow(
                hour_day_orders.T,
                labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
                title="Order Heatmap: Hour vs Day of Week",
                aspect="auto",
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig_hour_heatmap, use_container_width=True, key='operations_hour_heatmap')
        
        with col2:
            # Platform-specific peak hours
            platform_peak_hours = completed_df.groupby(['Platform', 'Hour']).size().reset_index(name='Orders')
            
            fig_platform_hours = px.line(
                platform_peak_hours,
                x='Hour',
                y='Orders',
                color='Platform',
                title="Peak Hours by Platform",
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            st.plotly_chart(fig_platform_hours, use_container_width=True, key='operations_platform_hours')
        
        # Operational efficiency metrics
        st.markdown("### 📊 Operational Efficiency Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate average processing metrics
            processing_metrics = pd.DataFrame({
                'Platform': completed_df['Platform'].unique(),
                'Avg Orders/Hour': [
                    len(completed_df[completed_df['Platform'] == p]) / 
                    (completed_df[completed_df['Platform'] == p]['Hour'].nunique() or 1)
                    for p in completed_df['Platform'].unique()
                ],
                'Peak Hour': [
                    completed_df[completed_df['Platform'] == p].groupby('Hour').size().idxmax()
                    if len(completed_df[completed_df['Platform'] == p]) > 0 else 0
                    for p in completed_df['Platform'].unique()
                ]
            })
            
            st.dataframe(processing_metrics, use_container_width=True, hide_index=True)
        
        with col2:
            # Day of week performance
            dow_performance = completed_df.groupby(['DayOfWeek', 'Platform']).agg({
                'Revenue': 'mean',
                'Order_ID': 'count'
            }).round(2)
            dow_performance.columns = ['Avg Revenue', 'Order Count']
            dow_performance = dow_performance.reset_index()
            
            # Sort by day order
            dow_performance['DayOfWeek'] = pd.Categorical(
                dow_performance['DayOfWeek'],
                categories=day_order,
                ordered=True
            )
            dow_performance = dow_performance.sort_values(['Platform', 'DayOfWeek'])
            
            st.dataframe(dow_performance.head(10), use_container_width=True, hide_index=True)
        
        # Store operational metrics
        st.markdown("### 🏪 Store Operational Metrics")
        
        store_ops = completed_df.groupby(['Store_Name', 'Platform']).agg({
            'Revenue': ['sum', 'mean'],
            'Order_ID': 'count',
            'Hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else 0
        }).round(2)
        store_ops.columns = ['Total Revenue', 'Avg Revenue', 'Order Count', 'Peak Hour']
        store_ops = store_ops.reset_index()
        store_ops = store_ops.sort_values('Total Revenue', ascending=False).head(15)
        
        st.dataframe(store_ops, use_container_width=True, hide_index=True)
    
    # TAB 5: GROWTH & TRENDS
    with tab5:
        st.markdown("### 📈 Growth Analysis & Forecasting")
        
        # Growth metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Month-over-month growth
            monthly_growth = completed_df.groupby(['Month', 'Platform'])['Revenue'].sum().reset_index()
            monthly_growth['Month'] = monthly_growth['Month'].astype(str)
            
            fig_mom_growth = px.line(
                monthly_growth,
                x='Month',
                y='Revenue',
                color='Platform',
                title="Month-over-Month Revenue Growth",
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            st.plotly_chart(fig_mom_growth, use_container_width=True, key='growth_mom_revenue')
        
        with col2:
            # Order growth
            monthly_order_growth = completed_df.groupby(['Month', 'Platform']).size().reset_index(name='Orders')
            monthly_order_growth['Month'] = monthly_order_growth['Month'].astype(str)
            
            fig_order_growth = px.line(
                monthly_order_growth,
                x='Month',
                y='Orders',
                color='Platform',
                title="Month-over-Month Order Growth",
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            st.plotly_chart(fig_order_growth, use_container_width=True, key='growth_mom_orders')
        
        # Market share trend
        st.markdown("### 📊 Market Share Evolution")
        
        market_share = completed_df.groupby(['Month', 'Platform'])['Revenue'].sum().reset_index()
        market_share['Total'] = market_share.groupby('Month')['Revenue'].transform('sum')
        market_share['Market_Share'] = (market_share['Revenue'] / market_share['Total'] * 100).round(2)
        market_share['Month'] = market_share['Month'].astype(str)
        
        fig_market_share = px.area(
            market_share,
            x='Month',
            y='Market_Share',
            color='Platform',
            title="Platform Market Share Over Time (%)",
            color_discrete_map=PLATFORM_COLORS
        )
        st.plotly_chart(fig_market_share, use_container_width=True, key='growth_market_share')
        
        # Growth rate comparison
        st.markdown("### 📈 Growth Rate Comparison")
        
        if len(monthly_revenue) > 1:
            growth_rates = []
            for platform in completed_df['Platform'].unique():
                platform_monthly = monthly_revenue[monthly_revenue['Platform'] == platform].copy()
                platform_monthly = platform_monthly.sort_values('Month')
                if len(platform_monthly) > 1:
                    platform_monthly['Growth_Rate'] = platform_monthly['Revenue'].pct_change() * 100
                    platform_monthly['Platform'] = platform
                    growth_rates.append(platform_monthly[['Month', 'Platform', 'Growth_Rate']].iloc[1:])
            
            if growth_rates:
                growth_df = pd.concat(growth_rates, ignore_index=True)
                
                fig_growth_rate = px.bar(
                    growth_df,
                    x='Month',
                    y='Growth_Rate',
                    color='Platform',
                    title="Monthly Growth Rate by Platform (%)",
                    color_discrete_map=PLATFORM_COLORS,
                    barmode='group',
                    text='Growth_Rate'
                )
                fig_growth_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_growth_rate, use_container_width=True, key='growth_rate_comparison')
        
        # Trend analysis
        st.markdown("### 📊 Trend Analysis")
        
        # Calculate moving averages
        ma_window = 7  # 7-day moving average
        daily_revenue_ma = daily_revenue.copy()
        
        for platform in daily_revenue_ma['Platform'].unique():
            mask = daily_revenue_ma['Platform'] == platform
            daily_revenue_ma.loc[mask, 'MA7'] = daily_revenue_ma.loc[mask, 'Revenue'].rolling(window=ma_window, min_periods=1).mean()
        
        fig_trend = go.Figure()
        
        for platform in daily_revenue_ma['Platform'].unique():
            platform_data = daily_revenue_ma[daily_revenue_ma['Platform'] == platform]
            
            # Add actual revenue
            fig_trend.add_trace(go.Scatter(
                x=platform_data['Date'],
                y=platform_data['Revenue'],
                mode='lines',
                name=f'{platform} (Actual)',
                line=dict(color=PLATFORM_COLORS.get(platform, '#000000'), width=1, dash='dot'),
                opacity=0.5
            ))
            
            # Add moving average
            fig_trend.add_trace(go.Scatter(
                x=platform_data['Date'],
                y=platform_data['MA7'],
                mode='lines',
                name=f'{platform} (7-Day MA)',
                line=dict(color=PLATFORM_COLORS.get(platform, '#000000'), width=2)
            ))
        
        fig_trend.update_layout(
            title="Revenue Trend with 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True, key='growth_trend_analysis')
    
    # TAB 6: CUSTOMER ATTRIBUTION
    with tab6:
        st.markdown("### 🎯 Customer Attribution & Segmentation")
        
        # RFM Analysis
        st.markdown("### 💎 RFM Analysis")
        
        rfm_data = calculate_rfm_scores(completed_df)
        
        if not rfm_data.empty:
            # RFM segment distribution
            segment_counts = rfm_data['Segment'].value_counts()
            
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Segmentation Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_segments, use_container_width=True, key='attribution_rfm_segments')
            
            # RFM metrics by segment
            segment_metrics = rfm_data.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rfm_scatter = px.scatter(
                    rfm_data,
                    x='Frequency',
                    y='Monetary',
                    color='Segment',
                    size='Recency',
                    title="RFM Customer Segmentation",
                    labels={
                        'Frequency': 'Purchase Frequency',
                        'Monetary': 'Total Spend ($)'
                    },
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_rfm_scatter, use_container_width=True, key='attribution_rfm_scatter')
            
            with col2:
                st.markdown("#### Segment Metrics")
                st.dataframe(segment_metrics, use_container_width=True)
        
        # Customer Lifetime Value Analysis
        st.markdown("### 💰 Customer Lifetime Value Analysis")
        
        # Simulate CLV calculation
        clv_data = completed_df.groupby('Order_ID').agg({
            'Revenue': 'sum',
            'Date': 'first',
            'Platform': 'first'
        }).reset_index()
        
        clv_data['Days_Since_First'] = (completed_df['Date'].max() - clv_data['Date']).dt.days
        clv_data['Estimated_CLV'] = clv_data['Revenue'] * (365 / (clv_data['Days_Since_First'] + 1))
        
        # CLV by platform
        clv_by_platform = clv_data.groupby('Platform')['Estimated_CLV'].mean()
        
        fig_clv = px.bar(
            x=clv_by_platform.index,
            y=clv_by_platform.values,
            color=clv_by_platform.index,
            title="Average Customer Lifetime Value by Platform",
            color_discrete_map=PLATFORM_COLORS,
            text=clv_by_platform.values
        )
        fig_clv.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_clv.update_layout(showlegend=False, yaxis_title="CLV ($)")
        st.plotly_chart(fig_clv, use_container_width=True, key='attribution_clv_platform')
        
        # Behavioral segmentation
        st.markdown("### 🎯 Behavioral Segmentation")
        
        # Create behavioral segments based on order patterns
        behavioral_segments = completed_df.groupby('Order_ID').agg({
            'Revenue': ['sum', 'mean'],
            'Platform': 'first',
            'Hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else 0,
            'DayOfWeek': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        
        behavioral_segments.columns = ['Customer_ID', 'Total_Spend', 'AOV', 'Preferred_Platform', 'Preferred_Hour', 'Preferred_Day']
        
        # Categorize customers
        behavioral_segments['Spending_Category'] = pd.qcut(
            behavioral_segments['Total_Spend'],
            q=3,
            labels=['Low Spender', 'Medium Spender', 'High Spender']
        )
        
        # Visualize behavioral segments
        fig_behavioral = px.sunburst(
            behavioral_segments,
            path=['Preferred_Platform', 'Spending_Category', 'Preferred_Day'],
            values='Total_Spend',
            title="Customer Behavioral Segmentation Hierarchy",
            color='Total_Spend',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_behavioral, use_container_width=True, key='attribution_behavioral_sunburst')
        
        # Customer distribution by platform and spending
        platform_spending = behavioral_segments.groupby(['Preferred_Platform', 'Spending_Category']).size().reset_index(name='Customer_Count')
        
        fig_platform_spending = px.bar(
            platform_spending,
            x='Preferred_Platform',
            y='Customer_Count',
            color='Spending_Category',
            title="Customer Distribution by Platform and Spending Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
            barmode='stack'
        )
        st.plotly_chart(fig_platform_spending, use_container_width=True, key='attribution_platform_spending')
    
    # TAB 7: RETENTION & CHURN
    with tab7:
        st.markdown("### 🔄 Retention & Churn Analysis")
        
        # Calculate retention metrics
        cohort_data, monthly_active = calculate_retention_metrics(completed_df)
        
        # Monthly active customers
        if not monthly_active.empty:
            fig_active = px.line(
                monthly_active,
                x='Month',
                y='Active_Customers',
                color='Platform',
                title="Monthly Active Customers by Platform",
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            fig_active.update_xaxis(title="Month")
            fig_active.update_yaxis(title="Active Customers")
            st.plotly_chart(fig_active, use_container_width=True, key='retention_monthly_active')
        
        # Cohort analysis
        st.markdown("### 📊 Cohort Analysis")
        
        if not cohort_data.empty:
            # Create cohort retention table
            cohort_pivot = cohort_data.pivot_table(
                index='Cohort',
                columns='Platform',
                values='Customers',
                fill_value=0
            )
            
            fig_cohort = px.imshow(
                cohort_pivot,
                title="Customer Cohorts by Platform",
                labels=dict(x="Platform", y="Cohort", color="Customers"),
                aspect="auto",
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig_cohort, use_container_width=True, key='retention_cohort_heatmap')
        
        # Churn analysis
        st.markdown("### 📉 Churn Analysis")
        
        # Calculate churn rate (simplified - based on order frequency)
        order_frequency = completed_df.groupby(['Order_ID', 'Platform']).agg({
            'Date': ['min', 'max', 'count']
        }).reset_index()
        order_frequency.columns = ['Customer_ID', 'Platform', 'First_Order', 'Last_Order', 'Order_Count']
        
        # Calculate days since last order
        order_frequency['Days_Since_Last'] = (completed_df['Date'].max() - order_frequency['Last_Order']).dt.days
        
        # Define churned customers (no order in last 30 days)
        churn_threshold = 30
        order_frequency['Is_Churned'] = order_frequency['Days_Since_Last'] > churn_threshold
        
        # Churn rate by platform
        churn_by_platform = order_frequency.groupby('Platform')['Is_Churned'].mean() * 100
        
        fig_churn = px.bar(
            x=churn_by_platform.index,
            y=churn_by_platform.values,
            color=churn_by_platform.index,
            title=f"Churn Rate by Platform (>{churn_threshold} days inactive)",
            color_discrete_map=PLATFORM_COLORS,
            text=churn_by_platform.values
        )
        fig_churn.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_churn.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig_churn, use_container_width=True, key='retention_churn_rate')
        
        # Retention curve
        st.markdown("### 📈 Retention Curve")
        
        # Calculate retention over time
        days_range = range(0, min(91, order_frequency['Days_Since_Last'].max() + 1), 7)
        retention_data = []
        
        for platform in order_frequency['Platform'].unique():
            platform_data = order_frequency[order_frequency['Platform'] == platform]
            for days in days_range:
                active_customers = len(platform_data[platform_data['Days_Since_Last'] <= days])
                total_customers = len(platform_data)
                retention_rate = (active_customers / total_customers * 100) if total_customers > 0 else 0
                retention_data.append({
                    'Platform': platform,
                    'Days': days,
                    'Retention_Rate': retention_rate
                })
        
        retention_df = pd.DataFrame(retention_data)
        
        fig_retention_curve = px.line(
            retention_df,
            x='Days',
            y='Retention_Rate',
            color='Platform',
            title="Customer Retention Curve",
            color_discrete_map=PLATFORM_COLORS,
            markers=True
        )
        fig_retention_curve.update_layout(
            xaxis_title="Days Since First Order",
            yaxis_title="Retention Rate (%)"
        )
        st.plotly_chart(fig_retention_curve, use_container_width=True, key='retention_curve')
        
        # Customer lifecycle stages
        st.markdown("### 🔄 Customer Lifecycle Stages")
        
        # Define lifecycle stages based on order patterns
        def define_lifecycle_stage(row):
            if row['Order_Count'] == 1 and row['Days_Since_Last'] <= 7:
                return 'New Customer'
            elif row['Order_Count'] > 1 and row['Days_Since_Last'] <= 14:
                return 'Active'
            elif row['Order_Count'] > 5 and row['Days_Since_Last'] <= 30:
                return 'Loyal'
            elif row['Days_Since_Last'] > 30 and row['Days_Since_Last'] <= 60:
                return 'At Risk'
            elif row['Days_Since_Last'] > 60:
                return 'Churned'
            else:
                return 'Regular'
        
        order_frequency['Lifecycle_Stage'] = order_frequency.apply(define_lifecycle_stage, axis=1)
        
        # Visualize lifecycle distribution
        lifecycle_dist = order_frequency.groupby(['Platform', 'Lifecycle_Stage']).size().reset_index(name='Customer_Count')
        
        fig_lifecycle = px.bar(
            lifecycle_dist,
            x='Platform',
            y='Customer_Count',
            color='Lifecycle_Stage',
            title="Customer Lifecycle Stage Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            barmode='stack'
        )
        st.plotly_chart(fig_lifecycle, use_container_width=True, key='retention_lifecycle_stages')
        
        # Reactivation opportunities
        st.markdown("### 🎯 Reactivation Opportunities")
        
        at_risk_customers = order_frequency[order_frequency['Lifecycle_Stage'].isin(['At Risk', 'Churned'])]
        
        if not at_risk_customers.empty:
            reactivation_summary = at_risk_customers.groupby('Platform').agg({
                'Customer_ID': 'count',
                'Order_Count': 'mean',
                'Days_Since_Last': 'mean'
            }).round(2)
            reactivation_summary.columns = ['At-Risk Customers', 'Avg Past Orders', 'Avg Days Inactive']
            
            st.dataframe(reactivation_summary, use_container_width=True)
            
            # Potential revenue recovery
            avg_order_value_by_platform = completed_df.groupby('Platform')['Revenue'].mean()
            reactivation_summary['Potential Revenue Recovery'] = (
                reactivation_summary['At-Risk Customers'] * 
                avg_order_value_by_platform * 2  # Assume 2 orders per reactivated customer
            ).round(2)
            
            fig_recovery = px.bar(
                x=reactivation_summary.index,
                y=reactivation_summary['Potential Revenue Recovery'],
                color=reactivation_summary.index,
                title="Potential Revenue Recovery from Reactivation",
                color_discrete_map=PLATFORM_COLORS,
                text=reactivation_summary['Potential Revenue Recovery']
            )
            fig_recovery.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_recovery.update_layout(showlegend=False, yaxis_title="Potential Revenue ($)")
            st.plotly_chart(fig_recovery, use_container_width=True, key='retention_revenue_recovery')
    
    # TAB 8: PLATFORM COMPARISON
    with tab8:
        st.markdown("### 📱 Comprehensive Platform Comparison")
        
        # Create comprehensive comparison metrics
        comparison_metrics = pd.DataFrame()
        
        for platform in completed_df['Platform'].unique():
            platform_data = completed_df[completed_df['Platform'] == platform]
            
            metrics = {
                'Platform': platform,
                'Total Orders': len(platform_data),
                'Total Revenue': platform_data['Revenue'].sum(),
                'Average Order Value': platform_data['Revenue'].mean(),
                'Median Order Value': platform_data['Revenue'].median(),
                'Revenue Std Dev': platform_data['Revenue'].std(),
                'Min Order': platform_data['Revenue'].min(),
                'Max Order': platform_data['Revenue'].max(),
                'Daily Avg Revenue': platform_data.groupby('Date')['Revenue'].sum().mean(),
                'Peak Hour': platform_data.groupby('Hour').size().idxmax() if len(platform_data) > 0 else 0,
                'Top Day': platform_data.groupby('DayOfWeek').size().idxmax() if len(platform_data) > 0 else 'N/A',
                'Active Days': platform_data['Date'].nunique(),
                'Unique Stores': platform_data['Store_Name'].nunique()
            }
            
            comparison_metrics = pd.concat([comparison_metrics, pd.DataFrame([metrics])], ignore_index=True)
        
        # Display comparison table
        st.markdown("### 📊 Key Performance Indicators")
        
        # Format the metrics for display
        formatted_metrics = comparison_metrics.copy()
        formatted_metrics['Total Revenue'] = formatted_metrics['Total Revenue'].apply(lambda x: f"${x:,.2f}")
        formatted_metrics['Average Order Value'] = formatted_metrics['Average Order Value'].apply(lambda x: f"${x:.2f}")
        formatted_metrics['Median Order Value'] = formatted_metrics['Median Order Value'].apply(lambda x: f"${x:.2f}")
        formatted_metrics['Revenue Std Dev'] = formatted_metrics['Revenue Std Dev'].apply(lambda x: f"${x:.2f}")
        formatted_metrics['Min Order'] = formatted_metrics['Min Order'].apply(lambda x: f"${x:.2f}")
        formatted_metrics['Max Order'] = formatted_metrics['Max Order'].apply(lambda x: f"${x:.2f}")
        formatted_metrics['Daily Avg Revenue'] = formatted_metrics['Daily Avg Revenue'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(formatted_metrics, use_container_width=True, hide_index=True)
        
        # Radar chart comparison
        st.markdown("### 🎯 Multi-Dimensional Platform Analysis")
        
        # Normalize metrics for radar chart
        radar_metrics = comparison_metrics[['Platform', 'Total Orders', 'Total Revenue', 
                                           'Average Order Value', 'Active Days', 'Unique Stores']].copy()
        
        # Normalize each metric to 0-100 scale
        for col in radar_metrics.columns[1:]:
            max_val = radar_metrics[col].max()
            if max_val > 0:
                radar_metrics[col] = (radar_metrics[col] / max_val * 100).round(2)
        
        fig_radar = go.Figure()
        
        for _, row in radar_metrics.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Total Orders'], row['Total Revenue'], row['Average Order Value'],
                   row['Active Days'], row['Unique Stores']],
                theta=['Total Orders', 'Total Revenue', 'AOV', 'Active Days', 'Unique Stores'],
                fill='toself',
                name=row['Platform'],
                line_color=PLATFORM_COLORS.get(row['Platform'], '#000000')
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Platform Performance Radar Chart (Normalized)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, key='comparison_radar_chart')
        
        # Platform efficiency comparison
        st.markdown("### ⚡ Efficiency Metrics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue per day
            revenue_per_day = comparison_metrics[['Platform', 'Daily Avg Revenue']].copy()
            
            fig_rpd = px.bar(
                revenue_per_day,
                x='Platform',
                y='Daily Avg Revenue',
                color='Platform',
                title="Average Daily Revenue by Platform",
                color_discrete_map=PLATFORM_COLORS,
                text='Daily Avg Revenue'
            )
            fig_rpd.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_rpd.update_layout(showlegend=False, yaxis_title="Daily Revenue ($)")
            st.plotly_chart(fig_rpd, use_container_width=True, key='comparison_daily_revenue')
        
        with col2:
            # Orders per active day
            orders_per_day = comparison_metrics.copy()
            orders_per_day['Orders per Day'] = orders_per_day['Total Orders'] / orders_per_day['Active Days']
            
            fig_opd = px.bar(
                orders_per_day,
                x='Platform',
                y='Orders per Day',
                color='Platform',
                title="Average Orders per Active Day",
                color_discrete_map=PLATFORM_COLORS,
                text='Orders per Day'
            )
            fig_opd.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_opd.update_layout(showlegend=False, yaxis_title="Orders per Day")
            st.plotly_chart(fig_opd, use_container_width=True, key='comparison_orders_per_day')
        
        # Platform strengths and weaknesses
        st.markdown("### 💪 Platform Strengths & Opportunities")
        
        strengths = []
        for platform in comparison_metrics['Platform'].values:
            platform_row = comparison_metrics[comparison_metrics['Platform'] == platform].iloc[0]
            
            strength_items = []
            opportunity_items = []
            
            # Identify strengths (where platform ranks #1)
            if platform_row['Total Revenue'] == comparison_metrics['Total Revenue'].max():
                strength_items.append("Highest Total Revenue")
            if platform_row['Average Order Value'] == comparison_metrics['Average Order Value'].max():
                strength_items.append("Highest AOV")
            if platform_row['Total Orders'] == comparison_metrics['Total Orders'].max():
                strength_items.append("Most Orders")
            if platform_row['Active Days'] == comparison_metrics['Active Days'].max():
                strength_items.append("Most Active Days")
            
            # Identify opportunities (where platform ranks last)
            if platform_row['Total Revenue'] == comparison_metrics['Total Revenue'].min():
                opportunity_items.append("Increase Revenue")
            if platform_row['Average Order Value'] == comparison_metrics['Average Order Value'].min():
                opportunity_items.append("Improve AOV")
            if platform_row['Total Orders'] == comparison_metrics['Total Orders'].min():
                opportunity_items.append("Boost Order Volume")
            
            strengths.append({
                'Platform': platform,
                'Strengths': ', '.join(strength_items) if strength_items else 'Balanced Performance',
                'Opportunities': ', '.join(opportunity_items) if opportunity_items else 'Maintain Performance'
            })
        
        strengths_df = pd.DataFrame(strengths)
        st.dataframe(strengths_df, use_container_width=True, hide_index=True)
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Analytics Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report", key="export_excel_button"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Platform_Summary', index=False)
                
                # Revenue analysis
                daily_revenue.to_excel(writer, sheet_name='Daily_Revenue', index=False)
                
                # Efficiency metrics
                if 'efficiency_metrics' in locals():
                    efficiency_metrics.to_excel(writer, sheet_name='Efficiency_Metrics')
                
                # RFM analysis
                if not rfm_data.empty:
                    rfm_data.to_excel(writer, sheet_name='RFM_Analysis', index=False)
                
                # Platform comparison
                comparison_metrics.to_excel(writer, sheet_name='Platform_Comparison', index=False)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name=f"luckin_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_report"
            )
    
    with col2:
        if st.button("📈 Generate CSV Data", key="export_csv_button"):
            csv_output = completed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Data",
                data=csv_output,
                file_name=f"luckin_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv_data"
            )
    
    with col3:
        if st.button("📄 Generate Summary Report", key="export_summary_button"):
            report = f"""
LUCKIN COFFEE MARKETING ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
=================
Total Orders: {total_orders:,}
Total Revenue: ${total_revenue:,.2f}
Average Order Value: ${avg_order_value:.2f}
Completion Rate: {completion_rate:.1f}%
Revenue Growth (MoM): {revenue_growth:.1f}%
Order Growth (MoM): {order_growth:.1f}%

PLATFORM BREAKDOWN
==================
{summary_df.to_string()}

TOP INSIGHTS
============
1. Highest revenue platform: {platform_revenue.idxmax()}
2. Most orders platform: {platform_orders.idxmax()}
3. Peak operating hour: {completed_df.groupby('Hour').size().idxmax()}:00
4. Best performing day: {completed_df.groupby('DayOfWeek')['Revenue'].sum().idxmax()}
5. Top performing store: {store_performance.index[0] if len(store_performance) > 0 else 'N/A'}

Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
"""
            st.download_button(
                label="📥 Download Summary Report",
                data=report,
                file_name=f"luckin_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_summary_report"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Luckin Coffee Marketing Analytics Dashboard v2.0</p>
            <p style='font-size: 0.9rem;'>Powered by Streamlit & Plotly | Data-Driven Insights for Growth</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
