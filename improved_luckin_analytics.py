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

# Data Processing Functions with complete fixes
@st.cache_data
def process_doordash_data(df):
    """Process DoorDash data with improved error handling"""
    try:
        processed = pd.DataFrame()
        
        # Core fields
        processed['Date'] = pd.to_datetime(df['时间戳本地日期'], errors='coerce')
        processed['Platform'] = 'DoorDash'
        processed['Revenue'] = pd.to_numeric(df['净总计'], errors='coerce')
        
        # Optional fields with safe access
        for col, new_col in [
            ('小计', 'Subtotal'),
            ('转交给商家的税款小计', 'Tax'),
            ('员工小费', 'Tips'),
            ('佣金', 'Commission'),
            ('营销费 |（包括任何适用税金）', 'Marketing_Fee')
        ]:
            if col in df.columns:
                processed[new_col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                processed[new_col] = 0
        
        # Process order status
        if '最终订单状态' in df.columns:
            processed['Is_Completed'] = df['最终订单状态'].astype(str).str.contains('Delivered|delivered', case=False, na=False, regex=True)
            processed['Is_Cancelled'] = df['最终订单状态'].astype(str).str.contains('Cancelled|cancelled', case=False, na=False, regex=True)
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information
        processed['Store_Name'] = df.get('店铺名称', 'Unknown').fillna('Unknown')
        processed['Store_ID'] = df.get('Store ID', 'Unknown').fillna('Unknown').astype(str)
        
        # Order ID for unique customer tracking
        if 'DoorDash 订单 ID' in df.columns:
            processed['Order_ID'] = df['DoorDash 订单 ID'].astype(str)
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_dd'
        
        # Time processing
        if '时间戳为本地时间' in df.columns:
            try:
                time_series = pd.to_datetime(df['时间戳为本地时间'], errors='coerce')
                processed['Hour'] = time_series.dt.hour.fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        # Add day and month info
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        
        # Clean data
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        
        return processed
    except Exception as e:
        st.error(f"DoorDash processing error: {e}")
        return pd.DataFrame()

@st.cache_data
def process_uber_data(df):
    """Process Uber data with fixed header handling"""
    try:
        # Check if the file has the two-row header structure
        # The first column would have the long Chinese description
        if 'Uber Eats 优食管理工具中显示的餐厅名称' in str(df.columns[0]):
            # The current headers are the description row
            # The actual headers are in the first data row (iloc[0])
            # The real data starts from iloc[1]
            
            # Get the actual column headers from the first row
            new_columns = []
            header_row = df.iloc[0] if len(df) > 0 else []
            
            for val in header_row:
                if pd.isna(val) or str(val) == 'nan':
                    new_columns.append('')
                else:
                    new_columns.append(str(val).strip())
            
            # Set the new column names
            df.columns = new_columns
            
            # Remove the header row and keep only actual data
            df = df.iloc[1:].reset_index(drop=True)
        
        # Clean empty column names
        df.columns = [col if col else f'col_{i}' for i, col in enumerate(df.columns)]
        
        processed = pd.DataFrame()
        
        # Process Date - the column might be named '订单日期'
        date_col = None
        for col in df.columns:
            if '订单日期' in col or '日期' in col:
                date_col = col
                break
        
        if date_col:
            # Handle date in various formats
            date_str = df[date_col].astype(str)
            # Remove any time components if present
            date_str = date_str.str.split(' ').str[0]
            # Try different date formats
            processed['Date'] = pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
            if processed['Date'].isna().all():
                processed['Date'] = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
            if processed['Date'].isna().all():
                processed['Date'] = pd.to_datetime(date_str, errors='coerce')
        else:
            processed['Date'] = pd.NaT
        
        processed['Platform'] = 'Uber'
        
        # Process Revenue - look for '收入总额' column
        revenue_col = None
        for col in df.columns:
            if '收入总额' in col or '收入' in col:
                revenue_col = col
                break
        
        if revenue_col:
            # Remove any spaces and handle the column properly
            revenue_str = df[revenue_col].astype(str).str.strip()
            processed['Revenue'] = pd.to_numeric(revenue_str, errors='coerce')
        else:
            processed['Revenue'] = 0
        
        # Process optional fields
        field_mappings = {
            '销售额（不含税费）': 'Subtotal',
            '销售额税费': 'Tax',
            '小费': 'Tips',
            '平台服务费': 'Commission',
            '营销调整额': 'Marketing_Fee'
        }
        
        for chinese_col, eng_col in field_mappings.items():
            found_col = None
            for col in df.columns:
                if chinese_col in col:
                    found_col = col
                    break
            
            if found_col:
                processed[eng_col] = pd.to_numeric(df[found_col].astype(str).str.strip(), errors='coerce').fillna(0)
            else:
                processed[eng_col] = 0
        
        # Process order status
        status_col = None
        for col in df.columns:
            if '订单状态' in col or '状态' in col:
                status_col = col
                break
        
        if status_col:
            status_str = df[status_col].astype(str)
            processed['Is_Completed'] = status_str.str.contains('已完成|completed', case=False, na=False, regex=True)
            processed['Is_Cancelled'] = status_str.str.contains('已取消|cancelled', case=False, na=False, regex=True)
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information
        store_col = None
        for col in df.columns:
            if '餐厅名称' in col or '餐厅' in col:
                store_col = col
                break
        
        if store_col:
            processed['Store_Name'] = df[store_col].fillna('Unknown').astype(str)
        else:
            processed['Store_Name'] = 'Unknown'
        
        # Store ID
        store_id_col = None
        for col in df.columns:
            if '餐厅号' in col:
                store_id_col = col
                break
        
        if store_id_col:
            processed['Store_ID'] = df[store_id_col].fillna('Unknown').astype(str)
        else:
            processed['Store_ID'] = 'Unknown'
        
        # Order ID
        order_col = None
        for col in df.columns:
            if '订单号' in col:
                order_col = col
                break
        
        if order_col:
            processed['Order_ID'] = df[order_col].fillna('').astype(str)
            # Replace empty order IDs with generated ones
            empty_mask = processed['Order_ID'] == ''
            processed.loc[empty_mask, 'Order_ID'] = pd.Series(range(empty_mask.sum())).astype(str) + '_uber'
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_uber'
        
        # Time processing
        time_col = None
        for col in df.columns:
            if '订单接受时间' in col or '接受时间' in col:
                time_col = col
                break
        
        if time_col:
            try:
                time_series = pd.to_datetime(df[time_col], errors='coerce')
                processed['Hour'] = time_series.dt.hour.fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        # Add day and month info
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        
        # Clean data - remove rows with invalid dates or zero/null revenue
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        processed = processed[processed['Revenue'] != 0]
        
        # Additional validation - remove rows where revenue is suspiciously low (likely headers or errors)
        processed = processed[abs(processed['Revenue']) > 0.01]
        
        return processed
        
    except Exception as e:
        st.error(f"Uber processing error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

@st.cache_data
def process_grubhub_data(df):
    """Process Grubhub data with fixed date handling"""
    try:
        processed = pd.DataFrame()
        
        # Handle transaction_date - it might be corrupted (showing as #######)
        if 'transaction_date' in df.columns:
            date_str = df['transaction_date'].astype(str)
            
            # Try to parse normal dates first
            processed['Date'] = pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
            
            # If all dates failed, try other formats
            if processed['Date'].isna().all():
                processed['Date'] = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
            
            # If still failing, check for Excel serial dates or corrupted values
            if processed['Date'].isna().sum() > len(df) * 0.5:  # More than 50% failed
                # Try to convert from Excel serial date
                try:
                    # Remove any # symbols
                    date_clean = date_str.str.replace('#', '', regex=False)
                    # Try to convert numeric values
                    date_numeric = pd.to_numeric(date_clean, errors='coerce')
                    # Excel serial date starts from 1900-01-01
                    processed['Date'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(date_numeric - 2, unit='D')
                except:
                    # If all else fails, use a date range based on order
                    st.warning("Grubhub dates corrupted - using estimated date range")
                    start_date = pd.to_datetime('2025-10-01')
                    # Distribute orders across October
                    date_range = pd.date_range(start=start_date, periods=len(df), freq='H')
                    processed['Date'] = date_range[:len(df)]
        else:
            # No date column - use current date
            processed['Date'] = pd.to_datetime('today').normalize()
        
        processed['Platform'] = 'Grubhub'
        
        # Revenue - try multiple possible columns
        if 'merchant_net_total' in df.columns:
            processed['Revenue'] = pd.to_numeric(df['merchant_net_total'], errors='coerce')
        elif 'merchant_total' in df.columns:
            processed['Revenue'] = pd.to_numeric(df['merchant_total'], errors='coerce')
        else:
            processed['Revenue'] = 0
        
        # Optional fields
        field_mappings = {
            'subtotal': 'Subtotal',
            'subtotal_sales_tax': 'Tax',
            'tip': 'Tips',
            'commission': 'Commission',
            'merchant_funded_promotion': 'Marketing_Fee'
        }
        
        for col, new_col in field_mappings.items():
            if col in df.columns:
                processed[new_col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                processed[new_col] = 0
        
        # Order status - be more lenient with Grubhub data
        if 'transaction_type' in df.columns:
            trans_type = df['transaction_type'].astype(str)
            # Mark as cancelled only if explicitly cancelled or refund
            processed['Is_Cancelled'] = trans_type.str.contains('Cancel|cancel|Refund|refund', case=False, na=False, regex=True)
            # Mark as completed unless it's cancelled
            processed['Is_Completed'] = ~processed['Is_Cancelled']
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information
        processed['Store_Name'] = df.get('store_name', 'Unknown').fillna('Unknown')
        processed['Store_ID'] = df.get('store_number', 'Unknown').fillna('Unknown').astype(str)
        
        # Order ID
        if 'order_number' in df.columns:
            processed['Order_ID'] = df['order_number'].astype(str)
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_gh'
        
        # Time processing
        if 'transaction_time_local' in df.columns:
            try:
                time_str = df['transaction_time_local'].astype(str)
                # Remove # symbols
                time_str = time_str.str.replace('#', '', regex=False)
                # Try to parse time
                time_series = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
                processed['Hour'] = time_series.dt.hour.fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        # Add day and month info
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        
        # Clean data - remove invalid entries
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        processed = processed[processed['Revenue'] != 0]
        
        return processed
        
    except Exception as e:
        st.error(f"Grubhub processing error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def main():
    # Header
    st.markdown("""
        <div class="luckin-header">
            <h1 style='margin:0; font-size:2.5rem;'>☕ Luckin Coffee - Advanced Marketing Analytics Dashboard</h1>
            <p style='margin:0.5rem 0 0 0; opacity:0.9;'>Comprehensive Multi-Platform Performance Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for data upload
    with st.sidebar:
        st.markdown("## 📊 Data Upload Center")
        
        # DoorDash Upload
        st.markdown("#### DoorDash CSV")
        doordash_file = st.file_uploader(
            "Drag and drop file here",
            type=['csv'],
            key="doordash_upload",
            help="Limit 200MB per file • CSV"
        )
        
        # Uber Upload
        st.markdown("#### Uber CSV")
        uber_file = st.file_uploader(
            "Drag and drop file here",
            type=['csv'],
            key="uber_upload",
            help="Limit 200MB per file • CSV"
        )
        
        # Grubhub Upload
        st.markdown("#### Grubhub CSV")
        grubhub_file = st.file_uploader(
            "Drag and drop file here",
            type=['csv'],
            key="grubhub_upload",
            help="Limit 200MB per file • CSV"
        )
        
        st.markdown("---")
        
        # Date filter
        st.markdown("### 📅 Date Range Filter")
        use_date_filter = st.checkbox("Apply date filter", value=False)
    
    # Process uploaded files
    all_data = []
    upload_status = []
    
    # Process DoorDash
    if doordash_file is not None:
        try:
            dd_df = pd.read_csv(doordash_file)
            dd_processed = process_doordash_data(dd_df)
            if not dd_processed.empty:
                all_data.append(dd_processed)
                completed_count = dd_processed['Is_Completed'].sum()
                upload_status.append(f"✅ DoorDash: {len(dd_processed)} orders loaded ({completed_count} completed)")
            else:
                upload_status.append(f"❌ DoorDash: No valid data found (Raw rows: {len(dd_df)})")
        except Exception as e:
            upload_status.append(f"❌ DoorDash Error: {str(e)[:50]}")

    # Process Uber
    if uber_file is not None:
        try:
            uber_df = pd.read_csv(uber_file)
            uber_processed = process_uber_data(uber_df)
            if not uber_processed.empty:
                all_data.append(uber_processed)
                completed_count = uber_processed['Is_Completed'].sum()
                upload_status.append(f"✅ Uber: {len(uber_processed)} orders loaded ({completed_count} completed)")
            else:
                upload_status.append(f"❌ Uber: No valid data found (Raw rows: {len(uber_df)})")
        except Exception as e:
            upload_status.append(f"❌ Uber Error: {str(e)[:50]}")
    
    # Process Grubhub
    if grubhub_file is not None:
        try:
            gh_df = pd.read_csv(grubhub_file)
            gh_processed = process_grubhub_data(gh_df)
            if not gh_processed.empty:
                all_data.append(gh_processed)
                completed_count = gh_processed['Is_Completed'].sum()
                upload_status.append(f"✅ Grubhub: {len(gh_processed)} orders loaded ({completed_count} completed)")
            else:
                upload_status.append(f"❌ Grubhub: No valid data found (Raw rows: {len(gh_df)})")
        except Exception as e:
            upload_status.append(f"❌ Grubhub Error: {str(e)[:50]}")
    
    # Display upload status
    if upload_status:
        with st.sidebar:
            st.markdown("### 📋 Upload Status")
            for status in upload_status:
                if "✅" in status:
                    st.success(status)
                else:
                    st.error(status)
    
    # Check if we have any data
    if not all_data:
        st.info("👋 Welcome! Please upload at least one CSV file from the sidebar to begin analysis.")
        st.markdown("""
            ### Getting Started:
            1. Upload your delivery platform CSV files (DoorDash, Uber, Grubhub)
            2. The dashboard will automatically process and display your analytics
            3. Use the tabs above to explore different insights
            
            ### Supported Formats:
            - **DoorDash**: Standard merchant portal export
            - **Uber Eats**: Revenue report export (Chinese or English)
            - **Grubhub**: Transaction history export
            
            ### Troubleshooting:
            - **Uber**: Make sure your CSV has the two-row header format
            - **Grubhub**: Date corruption is handled automatically
            - **All platforms**: Revenue must be non-zero to be included
        """)
        return
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Apply date filter if selected
    if use_date_filter and not df.empty:
        with st.sidebar:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_filter"
            )
            
            if len(date_range) == 2:
                df = df[(df['Date'].dt.date >= date_range[0]) & 
                       (df['Date'].dt.date <= date_range[1])]
    
    # Check if we still have data after filtering
    if df.empty:
        st.warning("No data available for the selected date range. Please adjust your filters.")
        return
    
    # Calculate metrics
    completed_df = df[df['Is_Completed'] == True].copy()
    
    # Key metrics
    total_orders = len(df)
    total_revenue = df['Revenue'].sum()
    avg_order_value = df['Revenue'].mean() if len(df) > 0 else 0
    completion_rate = (len(completed_df) / len(df) * 100) if len(df) > 0 else 0
    cancellation_rate = (df['Is_Cancelled'].sum() / len(df) * 100) if len(df) > 0 else 0
    
    # Platform metrics
    platform_revenue = df.groupby('Platform')['Revenue'].sum()
    platform_orders = df.groupby('Platform').size()
    
    # Time-based analysis
    daily_revenue = df.groupby('Date')['Revenue'].sum().reset_index()
    monthly_revenue = df.groupby('Month')['Revenue'].sum()
    
    # Calculate growth metrics
    if len(monthly_revenue) > 1:
        revenue_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100)
    else:
        revenue_growth = 0
    
    monthly_orders = df.groupby('Month').size()
    if len(monthly_orders) > 1:
        order_growth = ((monthly_orders.iloc[-1] - monthly_orders.iloc[-2]) / monthly_orders.iloc[-2] * 100)
    else:
        order_growth = 0
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "💰 Revenue Analytics", "🏆 Performance", 
        "🕐 Operations", "📈 Growth & Trends", "🎯 Customer Attribution",
        "🔄 Retention & Churn", "📱 Platform Comparison"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown("### 📊 Executive Dashboard")
        
        # Display data summary
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.info(f"📅 Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        with col_info2:
            st.info(f"📊 Total Records: {len(df):,}")
        with col_info3:
            st.info(f"🏪 Active Platforms: {', '.join(df['Platform'].unique())}")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Orders", f"{total_orders:,}")
        with col2:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col3:
            st.metric("Average Order Value", f"${avg_order_value:.2f}")
        with col4:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        with col5:
            st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
        
        # Platform distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Order distribution pie chart
            if not platform_orders.empty:
                fig_orders = px.pie(
                    values=platform_orders.values,
                    names=platform_orders.index,
                    title="Order Distribution by Platform",
                    color_discrete_map=PLATFORM_COLORS
                )
                st.plotly_chart(fig_orders, use_container_width=True)
        
        with col2:
            # Revenue distribution pie chart
            if not platform_revenue.empty:
                fig_revenue = px.pie(
                    values=platform_revenue.values,
                    names=platform_revenue.index,
                    title="Revenue by Platform",
                    color_discrete_map=PLATFORM_COLORS
                )
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Daily trend
        if not daily_revenue.empty:
            fig_daily = px.line(
                daily_revenue,
                x='Date',
                y='Revenue',
                title="Daily Revenue Trend",
                markers=True
            )
            fig_daily.update_layout(showlegend=False)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Platform summary table
        st.markdown("### 📋 Platform Summary")
        if not platform_revenue.empty:
            summary_df = pd.DataFrame({
                'Platform': platform_revenue.index,
                'Total Orders': platform_orders.values,
                'Total Revenue': platform_revenue.values,
                'Average Order Value': [
                    df[df['Platform'] == p]['Revenue'].mean() 
                    for p in platform_revenue.index
                ],
                'Completion Rate (%)': [
                    df[df['Platform'] == p]['Is_Completed'].mean() * 100 
                    for p in platform_revenue.index
                ]
            })
            
            # Format the summary dataframe
            summary_df['Total Revenue'] = summary_df['Total Revenue'].apply(lambda x: f"${x:,.2f}")
            summary_df['Average Order Value'] = summary_df['Average Order Value'].apply(lambda x: f"${x:.2f}")
            summary_df['Completion Rate (%)'] = summary_df['Completion Rate (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    # TAB 2: REVENUE ANALYTICS
    with tab2:
        st.markdown("### 💰 Revenue Deep Dive")
        
        # Revenue metrics by platform
        col1, col2, col3 = st.columns(3)
        
        for idx, platform in enumerate(platform_revenue.index):
            platform_data = df[df['Platform'] == platform]
            with [col1, col2, col3][idx % 3]:
                st.markdown(f"#### {platform}")
                st.metric("Revenue", f"${platform_revenue[platform]:,.2f}")
                st.metric("Orders", f"{platform_orders[platform]:,}")
                st.metric("AOV", f"${platform_data['Revenue'].mean():.2f}")
        
        # Revenue breakdown
        st.markdown("### 📊 Revenue Components")
        
        revenue_components = pd.DataFrame({
            'Platform': [],
            'Subtotal': [],
            'Tax': [],
            'Tips': [],
            'Commission': [],
            'Marketing Fee': []
        })
        
        for platform in df['Platform'].unique():
            platform_data = df[df['Platform'] == platform]
            new_row = pd.DataFrame({
                'Platform': [platform],
                'Subtotal': [platform_data['Subtotal'].sum()],
                'Tax': [platform_data['Tax'].sum()],
                'Tips': [platform_data['Tips'].sum()],
                'Commission': [platform_data['Commission'].sum()],
                'Marketing Fee': [platform_data['Marketing_Fee'].sum()]
            })
            revenue_components = pd.concat([revenue_components, new_row], ignore_index=True)
        
        if not revenue_components.empty:
            fig_components = px.bar(
                revenue_components.melt(id_vars=['Platform'], var_name='Component', value_name='Amount'),
                x='Platform',
                y='Amount',
                color='Component',
                title="Revenue Components by Platform",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Time-based revenue analysis
        st.markdown("### 📈 Revenue Trends")
        
        # Platform revenue over time
        platform_daily = df.groupby(['Date', 'Platform'])['Revenue'].sum().reset_index()
        
        if not platform_daily.empty:
            fig_platform_trend = px.line(
                platform_daily,
                x='Date',
                y='Revenue',
                color='Platform',
                title="Daily Revenue by Platform",
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            st.plotly_chart(fig_platform_trend, use_container_width=True)
    
    # TAB 3: PERFORMANCE
    with tab3:
        st.markdown("### 🏆 Performance Metrics")
        
        # Store performance
        store_performance = df.groupby('Store_Name')['Revenue'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
        
        # Top performing stores
        if not store_performance.empty:
            st.markdown("#### 🏪 Top Performing Stores")
            top_stores = store_performance.head(10).copy()
            top_stores.columns = ['Total Revenue', 'Order Count', 'Average Order Value']
            top_stores['Total Revenue'] = top_stores['Total Revenue'].apply(lambda x: f"${x:,.2f}")
            top_stores['Average Order Value'] = top_stores['Average Order Value'].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(top_stores, use_container_width=True)
        
        # Performance by day of week
        st.markdown("#### 📅 Day of Week Performance")
        
        if 'DayOfWeek' in df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_performance = df.groupby('DayOfWeek')['Revenue'].agg(['sum', 'count', 'mean'])
            
            # Reorder days properly
            available_days = [day for day in day_order if day in dow_performance.index]
            if available_days:
                dow_performance = dow_performance.reindex(available_days)
                
                fig_dow = px.bar(
                    x=dow_performance.index,
                    y=dow_performance['sum'],
                    title="Revenue by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Total Revenue'},
                    color=dow_performance['sum'],
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_dow, use_container_width=True)
    
    # TAB 4: OPERATIONS
    with tab4:
        st.markdown("### 🕐 Operational Insights")
        
        # Hourly analysis
        if 'Hour' in df.columns:
            hourly_orders = df.groupby('Hour').size()
            hourly_revenue = df.groupby('Hour')['Revenue'].sum()
            
            if not hourly_orders.empty:
                fig_hourly = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Orders by Hour', 'Revenue by Hour')
                )
                
                fig_hourly.add_trace(
                    go.Bar(x=hourly_orders.index, y=hourly_orders.values, name='Orders'),
                    row=1, col=1
                )
                
                fig_hourly.add_trace(
                    go.Bar(x=hourly_revenue.index, y=hourly_revenue.values, name='Revenue'),
                    row=1, col=2
                )
                
                fig_hourly.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Order completion analysis
        st.markdown("#### ✅ Order Completion Analysis")
        
        completion_by_platform = df.groupby('Platform')['Is_Completed'].mean() * 100
        cancellation_by_platform = df.groupby('Platform')['Is_Cancelled'].mean() * 100
        
        if not completion_by_platform.empty:
            fig_completion = go.Figure()
            fig_completion.add_trace(go.Bar(
                name='Completion Rate',
                x=completion_by_platform.index,
                y=completion_by_platform.values,
                marker_color='green'
            ))
            fig_completion.add_trace(go.Bar(
                name='Cancellation Rate',
                x=cancellation_by_platform.index,
                y=cancellation_by_platform.values,
                marker_color='red'
            ))
            
            fig_completion.update_layout(
                title="Order Status by Platform",
                yaxis_title="Percentage (%)",
                barmode='group'
            )
            st.plotly_chart(fig_completion, use_container_width=True)
    
    # TAB 5: GROWTH & TRENDS
    with tab5:
        st.markdown("### 📈 Growth Analysis")
        
        # Monthly growth
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Revenue Growth (MoM)",
                f"{revenue_growth:+.1f}%",
                delta=f"${monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]:,.2f}" if len(monthly_revenue) > 1 else "N/A"
            )
        
        with col2:
            st.metric(
                "Order Growth (MoM)",
                f"{order_growth:+.1f}%",
                delta=f"{monthly_orders.iloc[-1] - monthly_orders.iloc[-2]:+,} orders" if len(monthly_orders) > 1 else "N/A"
            )
        
        # Growth trend chart
        if len(monthly_revenue) > 0:
            # Format month labels to be more readable (e.g., "Oct 2025")
            month_labels = [period.strftime('%b %Y') for period in monthly_revenue.index]

            fig_growth = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly Revenue Trend', 'Monthly Order Volume'),
                vertical_spacing=0.1
            )

            fig_growth.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_revenue.values,
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ),
                row=1, col=1
            )

            fig_growth.add_trace(
                go.Scatter(
                    x=month_labels,
                    y=monthly_orders.values,
                    mode='lines+markers',
                    name='Orders',
                    line=dict(color='green', width=3),
                    marker=dict(size=10)
                ),
                row=2, col=1
            )

            fig_growth.update_layout(height=600, showlegend=False)
            fig_growth.update_xaxes(title_text="Month", row=2, col=1)
            fig_growth.update_yaxes(title_text="Revenue ($)", row=1, col=1)
            fig_growth.update_yaxes(title_text="Number of Orders", row=2, col=1)

            st.plotly_chart(fig_growth, use_container_width=True)
    
    # TAB 6: CUSTOMER ATTRIBUTION
    with tab6:
        st.markdown("### 🎯 Customer Analysis")
        
        # Calculate customer metrics
        customer_orders = df.groupby('Order_ID')['Revenue'].sum()
        
        if not customer_orders.empty:
            # Customer segmentation
            st.markdown("#### 💎 Customer Value Segmentation")
            
            # Define segments based on order value
            def segment_customer(value):
                if value >= 50:
                    return 'Premium'
                elif value >= 20:
                    return 'Regular'
                else:
                    return 'Budget'
            
            customer_segments = customer_orders.apply(segment_customer).value_counts()
            
            fig_segments = px.pie(
                values=customer_segments.values,
                names=customer_segments.index,
                title="Customer Segments by Order Value",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_segments, use_container_width=True)
            
            # Order value distribution
            st.markdown("#### 📊 Order Value Distribution")
            
            fig_dist = px.histogram(
                df,
                x='Revenue',
                nbins=50,
                title="Order Value Distribution",
                labels={'Revenue': 'Order Value ($)', 'count': 'Number of Orders'},
                color_discrete_sequence=['blue']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # TAB 7: RETENTION & CHURN
    with tab7:
        st.markdown("### 🔄 Retention & Churn Analysis")
        
        # Monthly order trends
        st.markdown("#### 📊 Order Volume Trends")
        
        # Create monthly analysis
        df['OrderMonth'] = df['Date'].dt.to_period('M')
        
        # Calculate retention metrics
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_order_trend = df.groupby('OrderMonth').size()

            if not monthly_order_trend.empty:
                # Format month labels to be more readable (e.g., "Oct 2025")
                month_labels = [period.strftime('%b %Y') for period in monthly_order_trend.index]

                fig_monthly = px.bar(
                    x=month_labels,
                    y=monthly_order_trend.values,
                    title="Monthly Order Volume",
                    labels={'x': 'Month', 'y': 'Number of Orders'},
                    color=monthly_order_trend.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            monthly_revenue_trend = df.groupby('OrderMonth')['Revenue'].sum()

            if not monthly_revenue_trend.empty:
                # Format month labels to be more readable (e.g., "Oct 2025")
                month_labels = [period.strftime('%b %Y') for period in monthly_revenue_trend.index]

                fig_rev = px.bar(
                    x=month_labels,
                    y=monthly_revenue_trend.values,
                    title="Monthly Revenue",
                    labels={'x': 'Month', 'y': 'Revenue ($)'},
                    color=monthly_revenue_trend.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_rev, use_container_width=True)
        
        # Churn analysis
        st.markdown("#### 🚪 Customer Activity Analysis")
        
        # Calculate days since last order per customer
        current_date = df['Date'].max()
        last_order_date = df.groupby('Order_ID')['Date'].max()
        days_since_order = (current_date - last_order_date).dt.days
        
        # Activity distribution
        if not days_since_order.empty:
            fig_activity = px.histogram(
                days_since_order,
                nbins=30,
                title="Days Since Last Order Distribution",
                labels={'value': 'Days Since Last Order', 'count': 'Number of Customers'}
            )
            st.plotly_chart(fig_activity, use_container_width=True)
    
    # TAB 8: PLATFORM COMPARISON
    with tab8:
        st.markdown("### 📱 Comprehensive Platform Comparison")
        
        if not completed_df.empty:
            # Create comprehensive comparison metrics
            comparison_metrics = pd.DataFrame()
            
            for platform in completed_df['Platform'].unique():
                platform_data = completed_df[completed_df['Platform'] == platform]
                
                metrics_dict = {
                    'Platform': platform,
                    'Total Orders': len(platform_data),
                    'Total Revenue': platform_data['Revenue'].sum(),
                    'Average Order Value': platform_data['Revenue'].mean(),
                    'Median Order Value': platform_data['Revenue'].median(),
                    'Revenue Std Dev': platform_data['Revenue'].std(),
                    'Min Order': platform_data['Revenue'].min(),
                    'Max Order': platform_data['Revenue'].max(),
                    'Daily Avg Revenue': platform_data.groupby('Date')['Revenue'].sum().mean(),
                    'Active Days': platform_data['Date'].nunique(),
                    'Unique Stores': platform_data['Store_Name'].nunique()
                }
                
                # Add peak hour if available
                if 'Hour' in platform_data.columns:
                    hour_counts = platform_data.groupby('Hour').size()
                    if not hour_counts.empty:
                        metrics_dict['Peak Hour'] = hour_counts.idxmax()
                    else:
                        metrics_dict['Peak Hour'] = 'N/A'
                
                # Add top day if available
                if 'DayOfWeek' in platform_data.columns:
                    day_counts = platform_data.groupby('DayOfWeek').size()
                    if not day_counts.empty:
                        metrics_dict['Top Day'] = day_counts.idxmax()
                    else:
                        metrics_dict['Top Day'] = 'N/A'
                
                new_row = pd.DataFrame([metrics_dict])
                comparison_metrics = pd.concat([comparison_metrics, new_row], ignore_index=True)
            
            # Display comparison table
            st.markdown("### 📊 Key Performance Indicators")
            
            if not comparison_metrics.empty:
                # Format the metrics for display
                formatted_metrics = comparison_metrics.copy()
                formatted_metrics['Total Revenue'] = formatted_metrics['Total Revenue'].apply(lambda x: f"${x:,.2f}")
                formatted_metrics['Average Order Value'] = formatted_metrics['Average Order Value'].apply(lambda x: f"${x:.2f}")
                formatted_metrics['Median Order Value'] = formatted_metrics['Median Order Value'].apply(lambda x: f"${x:.2f}")
                formatted_metrics['Revenue Std Dev'] = formatted_metrics['Revenue Std Dev'].apply(lambda x: f"${x:.2f}")
                formatted_metrics['Min Order'] = formatted_metrics['Min Order'].apply(lambda x: f"${x:.2f}")
                formatted_metrics['Max Order'] = formatted_metrics['Max Order'].apply(lambda x: f"${x:.2f}")
                formatted_metrics['Daily Avg Revenue'] = formatted_metrics['Daily Avg Revenue'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(formatted_metrics, hide_index=True, use_container_width=True)
                
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
                
                st.plotly_chart(fig_radar, use_container_width=True)
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Analytics Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                if 'summary_df' in locals():
                    summary_df.to_excel(writer, sheet_name='Platform_Summary', index=False)
                
                # Revenue analysis
                daily_revenue.to_excel(writer, sheet_name='Daily_Revenue', index=False)
                
                # Platform comparison
                if 'comparison_metrics' in locals():
                    comparison_metrics.to_excel(writer, sheet_name='Platform_Comparison', index=False)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name=f"luckin_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📈 Generate CSV Data"):
            csv_output = completed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Data",
                data=csv_output,
                file_name=f"luckin_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📄 Generate Summary Report"):
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
{platform_revenue.to_string() if not platform_revenue.empty else 'No data'}

TOP INSIGHTS
============
1. Highest revenue platform: {platform_revenue.idxmax() if not platform_revenue.empty else 'N/A'}
2. Most orders platform: {platform_orders.idxmax() if not platform_orders.empty else 'N/A'}
3. Top performing store: {store_performance.index[0] if len(store_performance) > 0 else 'N/A'}

Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
"""
            st.download_button(
                label="📥 Download Summary Report",
                data=report,
                file_name=f"luckin_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Luckin Coffee Marketing Analytics Dashboard v3.2</p>
            <p style='font-size: 0.9rem;'>All platforms working • Data updates in real-time</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
