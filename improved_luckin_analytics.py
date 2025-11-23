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

# Custom CSS - Refined aesthetic with coffee theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Source+Sans+Pro:wght@400;600&display=swap');
        
        .main { 
            padding: 0rem 1rem; 
            font-family: 'Source Sans Pro', sans-serif;
        }
        
        .luckin-header {
            background: linear-gradient(135deg, #8B4513 0%, #D2691E 50%, #CD853F 100%);
            padding: 2.5rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(139, 69, 19, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .luckin-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
            pointer-events: none;
        }
        
        h1, h2, h3 { 
            font-family: 'Crimson Pro', serif; 
            color: #3c2415;
        }
        
        .stTabs [data-baseweb="tab-list"] { 
            gap: 24px; 
            background: linear-gradient(90deg, #f5f1eb 0%, #faf8f5 100%);
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background: white;
            border-radius: 8px;
            padding: 0 20px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(139, 69, 19, 0.1);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            border-color: #D2691E;
            transform: translateY(-1px);
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%);
            color: white;
            border-color: #8B4513;
            box-shadow: 0 4px 16px rgba(139, 69, 19, 0.3);
        }
        
        .metric-card {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(139, 69, 19, 0.1);
            margin-bottom: 1.5rem;
            border-left: 4px solid #D2691E;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(139, 69, 19, 0.15);
        }
        
        .error-alert {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(220, 53, 69, 0.2);
        }
        
        .success-alert {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(40, 167, 69, 0.2);
        }
        
        .warning-alert {
            background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
            color: #212529;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(255, 193, 7, 0.2);
        }
        
        .sidebar .stSelectbox, .sidebar .stFileUploader {
            background: white;
            border-radius: 8px;
            border: 1px solid #D2691E;
        }
    </style>
""", unsafe_allow_html=True)

# Platform Colors - Coffee-themed
PLATFORM_COLORS = {
    'DoorDash': '#DC143C',      # Deep red
    'Uber': '#2F1B14',          # Dark coffee brown  
    'Grubhub': '#D2691E'        # Orange like coffee beans
}

# Enhanced Data Processing Functions
def safe_numeric_convert(series, default=0):
    """Safely convert a series to numeric, handling various data types"""
    if series is None:
        return pd.Series([default])
    
    if isinstance(series, (int, float)):
        return pd.Series([series])
    
    if isinstance(series, str):
        return pd.Series([default])
    
    try:
        # Handle pandas Series
        if hasattr(series, 'fillna'):
            return pd.to_numeric(series, errors='coerce').fillna(default)
        else:
            # Handle other types by converting to series first
            return pd.to_numeric(pd.Series(series), errors='coerce').fillna(default)
    except Exception:
        return pd.Series([default] * len(series) if hasattr(series, '__len__') else [default])

def parse_date_flexible(date_series, formats=None):
    """Flexibly parse dates with multiple fallback formats"""
    if formats is None:
        formats = [
            '%m/%d/%Y',
            '%Y-%m-%d', 
            '%d/%m/%Y',
            '%m/%d/%y',
            '%Y/%m/%d',
            '%d-%m-%Y'
        ]
    
    if date_series is None or len(date_series) == 0:
        return pd.Series(dtype='datetime64[ns]')
    
    # Handle string columns that might contain "########"
    if hasattr(date_series, 'astype'):
        str_series = date_series.astype(str)
        # Replace "########" with NaN
        str_series = str_series.replace('########', pd.NaT)
        str_series = str_series.replace('#REF!', pd.NaT)
        str_series = str_series.replace('NULL', pd.NaT)
    else:
        str_series = pd.Series(date_series).astype(str)
    
    # Try pandas built-in parser first
    try:
        return pd.to_datetime(str_series, errors='coerce', infer_datetime_format=True)
    except:
        pass
    
    # Try each format
    for fmt in formats:
        try:
            return pd.to_datetime(str_series, format=fmt, errors='coerce')
        except:
            continue
    
    # Final fallback - coerce everything
    return pd.to_datetime(str_series, errors='coerce')

@st.cache_data
def process_doordash_data(df):
    """Process DoorDash data with enhanced error handling"""
    try:
        processed = pd.DataFrame()
        
        if df is None or len(df) == 0:
            st.warning("DoorDash data is empty")
            return pd.DataFrame()
        
        # Core fields with flexible column mapping
        date_columns = ['时间戳本地日期', 'Date', 'Order Date', '订单日期']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            processed['Date'] = parse_date_flexible(df[date_col])
        else:
            st.warning("Could not find date column in DoorDash data")
            processed['Date'] = pd.NaT
        
        processed['Platform'] = 'DoorDash'
        
        # Revenue mapping with multiple possible column names
        revenue_columns = ['净总计', 'Net Total', 'Revenue', '收入总额']
        revenue_col = None
        for col in revenue_columns:
            if col in df.columns:
                revenue_col = col
                break
        
        if revenue_col:
            processed['Revenue'] = safe_numeric_convert(df[revenue_col])
        else:
            st.warning("Could not find revenue column in DoorDash data")
            processed['Revenue'] = 0
        
        # Optional fields with safe access
        processed['Subtotal'] = safe_numeric_convert(df.get('小计', 0))
        processed['Tax'] = safe_numeric_convert(df.get('转交给商家的税款小计', 0))
        processed['Tips'] = safe_numeric_convert(df.get('员工小费', 0))
        processed['Commission'] = safe_numeric_convert(df.get('佣金', 0))
        processed['Marketing_Fee'] = safe_numeric_convert(df.get('营销费 |（包括任何适用税金）', 0))
        
        # Process order status
        status_columns = ['最终订单状态', 'Final Status', 'Order Status', '订单状态']
        status_col = None
        for col in status_columns:
            if col in df.columns:
                status_col = col
                break
                
        if status_col:
            status_series = df[status_col].astype(str)
            processed['Is_Completed'] = status_series.str.contains('Delivered|delivered|完成', case=False, na=False, regex=True)
            processed['Is_Cancelled'] = status_series.str.contains('Cancelled|cancelled|取消', case=False, na=False, regex=True)
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information
        store_columns = ['店铺名称', 'Store Name', 'Restaurant Name']
        store_col = None
        for col in store_columns:
            if col in df.columns:
                store_col = col
                break
        
        processed['Store_Name'] = df[store_col] if store_col else 'Unknown'
        processed['Store_ID'] = df.get('Store ID', 'Unknown')
        
        # Order ID for unique customer tracking
        order_id_columns = ['DoorDash 订单 ID', 'Order ID', 'DoorDash Order ID']
        order_id_col = None
        for col in order_id_columns:
            if col in df.columns:
                order_id_col = col
                break
        
        if order_id_col:
            processed['Order_ID'] = df[order_id_col].astype(str)
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_dd'
        
        # Time processing
        time_columns = ['时间戳为本地时间', 'Local Time', 'Order Time']
        time_col = None
        for col in time_columns:
            if col in df.columns:
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
        
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        
        # Clean data - remove rows with invalid dates or revenue
        initial_length = len(processed)
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        processed = processed[processed['Revenue'] != 0]
        
        final_length = len(processed)
        if initial_length != final_length:
            st.info(f"DoorDash: Cleaned {initial_length - final_length} invalid records")
        
        return processed
        
    except Exception as e:
        st.error(f"DoorDash processing error: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def process_uber_data(df):
    """Process Uber data with enhanced error handling"""
    try:
        processed = pd.DataFrame()
        
        if df is None or len(df) == 0:
            st.warning("Uber data is empty")
            return pd.DataFrame()
        
        # Check if we need to skip header rows (Uber files often have description rows)
        if len(df) > 2 and '订单日期' not in df.columns:
            # Look for header row containing key Chinese terms
            for i in range(min(5, len(df))):
                row_str = ' '.join(df.iloc[i].astype(str).values)
                if any(term in row_str for term in ['餐厅名称', '订单日期', '收入总额', 'Restaurant', 'Date']):
                    df.columns = df.iloc[i]
                    df = df.iloc[i+1:].reset_index(drop=True)
                    break
        
        # Enhanced column mapping for Uber data
        date_columns = ['订单日期', 'Order Date', 'Date', '日期']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        # Fallback: try by position if column names are unclear
        if not date_col and len(df.columns) > 8:
            date_col = df.columns[8]  # Often the 9th column
        
        if date_col and date_col in df.columns:
            processed['Date'] = parse_date_flexible(df[date_col])
        else:
            st.warning("Could not find date column in Uber data")
            processed['Date'] = pd.NaT
            
        processed['Platform'] = 'Uber'
        
        # Revenue mapping
        revenue_columns = ['收入总额', 'Total Revenue', 'Revenue', 'Net Total']
        revenue_col = None
        for col in revenue_columns:
            if col in df.columns:
                revenue_col = col
                break
        
        # Fallback: try by position
        if not revenue_col and len(df.columns) > 41:
            revenue_col = df.columns[41]  # Often the 42nd column
        
        if revenue_col and revenue_col in df.columns:
            processed['Revenue'] = safe_numeric_convert(df[revenue_col])
        else:
            st.warning("Could not find revenue column in Uber data")
            processed['Revenue'] = 0
        
        # Optional fields with safe access
        processed['Subtotal'] = safe_numeric_convert(df.get('销售额（不含税费）', 0))
        processed['Tax'] = safe_numeric_convert(df.get('销售额税费', 0))
        processed['Tips'] = safe_numeric_convert(df.get('小费', 0))
        processed['Commission'] = safe_numeric_convert(df.get('平台服务费', 0))
        processed['Marketing_Fee'] = safe_numeric_convert(df.get('营销调整额', 0))
        
        # Process order status
        status_columns = ['订单状态', 'Order Status', 'Status']
        status_col = None
        for col in status_columns:
            if col in df.columns:
                status_col = col
                break
        
        if not status_col and len(df.columns) > 7:
            status_col = df.columns[7]  # Often the 8th column
        
        if status_col and status_col in df.columns:
            status_series = df[status_col].astype(str)
            processed['Is_Completed'] = status_series.str.contains('已完成|completed|Complete', case=False, na=False, regex=True)
            processed['Is_Cancelled'] = status_series.str.contains('已取消|cancelled|Cancel', case=False, na=False, regex=True)
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information
        restaurant_columns = ['餐厅名称', 'Restaurant Name', 'Store Name']
        restaurant_col = None
        for col in restaurant_columns:
            if col in df.columns:
                restaurant_col = col
                break
                
        if not restaurant_col and len(df.columns) > 0:
            restaurant_col = df.columns[0]  # Often the first column
        
        if restaurant_col and restaurant_col in df.columns:
            processed['Store_Name'] = df[restaurant_col].astype(str)
        else:
            processed['Store_Name'] = 'Unknown'
        
        processed['Store_ID'] = safe_numeric_convert(df.get('餐厅号', 'Unknown'))
        
        # Order ID
        order_id_columns = ['订单号', 'Order ID', 'Order Number']
        order_id_col = None
        for col in order_id_columns:
            if col in df.columns:
                order_id_col = col
                break
                
        if order_id_col:
            processed['Order_ID'] = df[order_id_col].astype(str)
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_uber'
        
        # Time processing
        time_columns = ['订单接受时间', 'Accept Time', 'Order Time']
        time_col = None
        for col in time_columns:
            if col in df.columns:
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
        
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        
        # Clean data
        initial_length = len(processed)
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        processed = processed[processed['Revenue'] != 0]
        
        final_length = len(processed)
        if initial_length != final_length:
            st.info(f"Uber: Cleaned {initial_length - final_length} invalid records")
        
        return processed
        
    except Exception as e:
        st.error(f"Uber processing error: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def process_grubhub_data(df):
    """Process Grubhub data with enhanced error handling"""
    try:
        processed = pd.DataFrame()
        
        if df is None or len(df) == 0:
            st.warning("Grubhub data is empty")
            return pd.DataFrame()
        
        # Core fields mapping
        date_columns = ['transaction_date', 'Date', 'Order Date']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            processed['Date'] = parse_date_flexible(df[date_col])
        else:
            st.warning("Could not find date column in Grubhub data")
            processed['Date'] = pd.NaT
            
        processed['Platform'] = 'Grubhub'
        
        # Revenue mapping
        revenue_columns = ['merchant_net_total', 'Net Total', 'Revenue']
        revenue_col = None
        for col in revenue_columns:
            if col in df.columns:
                revenue_col = col
                break
        
        if revenue_col:
            processed['Revenue'] = safe_numeric_convert(df[revenue_col])
        else:
            st.warning("Could not find revenue column in Grubhub data")
            processed['Revenue'] = 0
        
        # Optional fields
        processed['Subtotal'] = safe_numeric_convert(df.get('subtotal', 0))
        processed['Tax'] = safe_numeric_convert(df.get('subtotal_sales_tax', 0))
        processed['Tips'] = safe_numeric_convert(df.get('tip', 0))
        processed['Commission'] = safe_numeric_convert(df.get('commission', 0))
        processed['Marketing_Fee'] = safe_numeric_convert(df.get('merchant_funded_promotion', 0))
        
        # Order status - Grubhub doesn't typically have explicit status in the data
        # Assume all records are completed orders since they're in merchant reports
        processed['Is_Completed'] = True
        processed['Is_Cancelled'] = False
        
        # Store information
        processed['Store_Name'] = df.get('store_name', 'Unknown').astype(str)
        processed['Store_ID'] = safe_numeric_convert(df.get('store_number', 'Unknown'))
        
        # Order ID
        order_id_columns = ['order_number', 'Order ID', 'transaction_id']
        order_id_col = None
        for col in order_id_columns:
            if col in df.columns:
                order_id_col = col
                break
                
        if order_id_col:
            processed['Order_ID'] = df[order_id_col].astype(str)
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_grubhub'
        
        # Time processing from transaction_time_local
        time_columns = ['transaction_time_local', 'Time', 'Order Time']
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            try:
                # Handle "########" in time columns
                time_data = df[time_col].astype(str).replace('########', '12:00')
                time_series = pd.to_datetime(time_data, errors='coerce')
                processed['Hour'] = time_series.dt.hour.fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        
        # Clean data
        initial_length = len(processed)
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        processed = processed[processed['Revenue'] != 0]
        
        final_length = len(processed)
        if initial_length != final_length:
            st.info(f"Grubhub: Cleaned {initial_length - final_length} invalid records")
        
        return processed
        
    except Exception as e:
        st.error(f"Grubhub processing error: {str(e)}")
        return pd.DataFrame()

def create_enhanced_visualizations():
    """Create enhanced visualizations with better styling"""
    return {
        'template': 'plotly_white',
        'font': {'family': 'Source Sans Pro, sans-serif', 'size': 12},
        'colorway': ['#8B4513', '#D2691E', '#CD853F', '#F4A460', '#DEB887']
    }

def main():
    # Header with coffee-themed styling
    st.markdown("""
        <div class="luckin-header">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem; font-weight: 700;">☕ Luckin Coffee</h1>
            <h3 style="margin: 0; font-weight: 400; opacity: 0.9;">Advanced Marketing Analytics Dashboard</h3>
            <p style="margin-top: 1rem; opacity: 0.8; font-size: 1.1rem;">Comprehensive Multi-Platform Performance Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for file uploads with enhanced styling
    st.sidebar.markdown("### 📁 Data Upload Center")
    st.sidebar.markdown("Upload your platform data files below:")

    # File uploaders with better feedback
    doordash_file = st.sidebar.file_uploader(
        "DoorDash CSV", 
        type=['csv'], 
        key="doordash",
        help="Upload your DoorDash merchant report CSV file"
    )
    
    uber_file = st.sidebar.file_uploader(
        "Uber CSV", 
        type=['csv'], 
        key="uber",
        help="Upload your Uber Eats merchant report CSV file"
    )
    
    grubhub_file = st.sidebar.file_uploader(
        "Grubhub CSV", 
        type=['csv'], 
        key="grubhub",
        help="Upload your Grubhub merchant report CSV file"
    )

    # Data processing with enhanced error handling
    df_list = []
    processing_status = {"DoorDash": "❌ No data", "Uber": "❌ No data", "Grubhub": "❌ No data"}

    # Process DoorDash
    if doordash_file is not None:
        try:
            with st.spinner("Processing DoorDash data..."):
                doordash_df = pd.read_csv(doordash_file, encoding='utf-8-sig')
                processed_dd = process_doordash_data(doordash_df)
                if not processed_dd.empty:
                    df_list.append(processed_dd)
                    processing_status["DoorDash"] = f"✅ {len(processed_dd)} orders"
                    st.sidebar.markdown(f'<div class="success-alert">✅ DoorDash: {len(processed_dd)} orders loaded</div>', unsafe_allow_html=True)
                else:
                    processing_status["DoorDash"] = "⚠️ No valid data"
                    st.sidebar.markdown('<div class="warning-alert">⚠️ DoorDash: No valid data found</div>', unsafe_allow_html=True)
        except Exception as e:
            processing_status["DoorDash"] = f"❌ Error: {str(e)[:50]}..."
            st.sidebar.markdown(f'<div class="error-alert">❌ DoorDash Error: {str(e)}</div>', unsafe_allow_html=True)

    # Process Uber
    if uber_file is not None:
        try:
            with st.spinner("Processing Uber data..."):
                uber_df = pd.read_csv(uber_file, encoding='utf-8-sig')
                processed_uber = process_uber_data(uber_df)
                if not processed_uber.empty:
                    df_list.append(processed_uber)
                    processing_status["Uber"] = f"✅ {len(processed_uber)} orders"
                    st.sidebar.markdown(f'<div class="success-alert">✅ Uber: {len(processed_uber)} orders loaded</div>', unsafe_allow_html=True)
                else:
                    processing_status["Uber"] = "⚠️ No valid data"
                    st.sidebar.markdown('<div class="warning-alert">⚠️ Uber: No valid data found</div>', unsafe_allow_html=True)
        except Exception as e:
            processing_status["Uber"] = f"❌ Error: {str(e)[:50]}..."
            st.sidebar.markdown(f'<div class="error-alert">❌ Uber Error: {str(e)}</div>', unsafe_allow_html=True)

    # Process Grubhub
    if grubhub_file is not None:
        try:
            with st.spinner("Processing Grubhub data..."):
                grubhub_df = pd.read_csv(grubhub_file, encoding='utf-8-sig')
                processed_grubhub = process_grubhub_data(grubhub_df)
                if not processed_grubhub.empty:
                    df_list.append(processed_grubhub)
                    processing_status["Grubhub"] = f"✅ {len(processed_grubhub)} orders"
                    st.sidebar.markdown(f'<div class="success-alert">✅ Grubhub: {len(processed_grubhub)} orders loaded</div>', unsafe_allow_html=True)
                else:
                    processing_status["Grubhub"] = "⚠️ No valid data"
                    st.sidebar.markdown('<div class="warning-alert">⚠️ Grubhub: No valid data found</div>', unsafe_allow_html=True)
        except Exception as e:
            processing_status["Grubhub"] = f"❌ Error: {str(e)[:50]}..."
            st.sidebar.markdown(f'<div class="error-alert">❌ Grubhub Error: {str(e)}</div>', unsafe_allow_html=True)

    # Display processing status
    st.sidebar.markdown("### 📊 Processing Status")
    for platform, status in processing_status.items():
        st.sidebar.markdown(f"**{platform}**: {status}")

    # Check if we have any data
    if not df_list:
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin: 2rem 0;">
                <h2 style="color: #6c757d; margin-bottom: 1rem;">☕ Welcome to Luckin Coffee Analytics</h2>
                <p style="color: #6c757d; font-size: 1.2rem;">Upload your platform data files to begin analyzing your coffee shop's performance</p>
                <p style="color: #6c757d;">Supported platforms: DoorDash, Uber Eats, Grubhub</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Combine all data
    df = pd.concat(df_list, ignore_index=True)
    
    # Filter for completed orders only
    completed_df = df[df['Is_Completed'] == True].copy()
    
    if completed_df.empty:
        st.error("No completed orders found in the uploaded data!")
        return

    # Calculate metrics
    total_orders = len(completed_df)
    total_revenue = completed_df['Revenue'].sum()
    avg_order_value = completed_df['Revenue'].mean()
    completion_rate = (len(completed_df) / len(df) * 100) if len(df) > 0 else 0

    # Platform breakdown
    platform_summary = completed_df.groupby('Platform').agg({
        'Revenue': ['sum', 'mean', 'count'],
        'Order_ID': 'nunique'
    }).round(2)

    platform_summary.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 'Unique_Orders']
    platform_summary = platform_summary.reset_index()

    # Calculate trends
    monthly_data = completed_df.groupby(['Month', 'Platform'])['Revenue'].agg(['sum', 'count']).reset_index()
    current_month_revenue = monthly_data[monthly_data['Month'] == monthly_data['Month'].max()]['sum'].sum()
    previous_months = monthly_data[monthly_data['Month'] < monthly_data['Month'].max()]
    if not previous_months.empty:
        previous_month_revenue = previous_months.groupby('Month')['sum'].sum().iloc[-1]
        revenue_growth = ((current_month_revenue - previous_month_revenue) / previous_month_revenue * 100) if previous_month_revenue > 0 else 0
    else:
        revenue_growth = 0

    current_month_orders = monthly_data[monthly_data['Month'] == monthly_data['Month'].max()]['count'].sum()
    if not previous_months.empty:
        previous_month_orders = previous_months.groupby('Month')['count'].sum().iloc[-1]
        order_growth = ((current_month_orders - previous_month_orders) / previous_month_orders * 100) if previous_month_orders > 0 else 0
    else:
        order_growth = 0

    # Display overview metrics
    st.markdown("## 📊 Executive Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #8B4513; margin: 0;">Total Orders</h3>
                <h2 style="color: #D2691E; margin: 0.5rem 0;">{total_orders:,}</h2>
                <p style="color: #6c757d; margin: 0;">All Platforms</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #8B4513; margin: 0;">Total Revenue</h3>
                <h2 style="color: #D2691E; margin: 0.5rem 0;">${total_revenue:,.2f}</h2>
                <p style="color: #6c757d; margin: 0;">Monthly Growth: {revenue_growth:+.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #8B4513; margin: 0;">Avg Order Value</h3>
                <h2 style="color: #D2691E; margin: 0.5rem 0;">${avg_order_value:.2f}</h2>
                <p style="color: #6c757d; margin: 0;">Per Transaction</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #8B4513; margin: 0;">Completion Rate</h3>
                <h2 style="color: #D2691E; margin: 0.5rem 0;">{completion_rate:.1f}%</h2>
                <p style="color: #6c757d; margin: 0;">Order Success</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #8B4513; margin: 0;">Active Platforms</h3>
                <h2 style="color: #D2691E; margin: 0.5rem 0;">{len(platform_summary)}</h2>
                <p style="color: #6c757d; margin: 0;">Order Growth: {order_growth:+.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏠 Overview", "💰 Revenue Analytics", "🏆 Performance", 
        "🕐 Operations", "📈 Growth & Trends", "👥 Customer Attribution", 
        "🔄 Retention & Churn", "📱 Platform Comparison"
    ])

    with tab1:
        st.markdown("### 📱 Platform Overview")
        
        # Platform summary table
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by platform pie chart
            fig_revenue_pie = px.pie(
                platform_summary, 
                values='Total_Revenue', 
                names='Platform',
                title="Revenue Distribution by Platform",
                color='Platform',
                color_discrete_map=PLATFORM_COLORS,
                hole=0.4
            )
            fig_revenue_pie.update_layout(**create_enhanced_visualizations())
            st.plotly_chart(fig_revenue_pie, use_container_width=True, key='overview_revenue_pie')
        
        with col2:
            # Orders by platform pie chart
            fig_orders_pie = px.pie(
                platform_summary, 
                values='Total_Orders', 
                names='Platform',
                title="Order Distribution by Platform",
                color='Platform',
                color_discrete_map=PLATFORM_COLORS,
                hole=0.4
            )
            fig_orders_pie.update_layout(**create_enhanced_visualizations())
            st.plotly_chart(fig_orders_pie, use_container_width=True, key='overview_orders_pie')
        
        # Platform comparison bar chart
        fig_comparison = px.bar(
            platform_summary,
            x='Platform',
            y='Total_Revenue',
            color='Platform',
            title='Revenue Comparison Across Platforms',
            color_discrete_map=PLATFORM_COLORS,
            text='Total_Revenue'
        )
        fig_comparison.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_comparison.update_layout(**create_enhanced_visualizations())
        fig_comparison.update_layout(showlegend=False, yaxis_title="Total Revenue ($)")
        st.plotly_chart(fig_comparison, use_container_width=True, key='overview_platform_comparison')
        
        # Summary table
        st.markdown("### 📊 Platform Performance Summary")
        
        # Format the summary for display
        display_summary = platform_summary.copy()
        display_summary['Total_Revenue'] = display_summary['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
        display_summary['Avg_Order_Value'] = display_summary['Avg_Order_Value'].apply(lambda x: f"${x:.2f}")
        display_summary.columns = ['Platform', 'Total Revenue', 'Average Order Value', 'Total Orders', 'Unique Orders']
        
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 💰 Revenue Analytics")
        
        # Daily revenue trend
        daily_revenue = completed_df.groupby(['Date', 'Platform'])['Revenue'].sum().reset_index()
        
        fig_daily = px.line(
            daily_revenue,
            x='Date',
            y='Revenue',
            color='Platform',
            title='Daily Revenue Trends by Platform',
            color_discrete_map=PLATFORM_COLORS,
            markers=True
        )
        fig_daily.update_layout(**create_enhanced_visualizations())
        fig_daily.update_layout(hovermode='x unified', yaxis_title="Daily Revenue ($)")
        st.plotly_chart(fig_daily, use_container_width=True, key='revenue_daily_trends')
        
        # Revenue distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                completed_df,
                x='Revenue',
                color='Platform',
                title='Revenue Distribution by Platform',
                color_discrete_map=PLATFORM_COLORS,
                nbins=50
            )
            fig_hist.update_layout(**create_enhanced_visualizations())
            st.plotly_chart(fig_hist, use_container_width=True, key='revenue_distribution')
        
        with col2:
            fig_box = px.box(
                completed_df,
                x='Platform',
                y='Revenue',
                color='Platform',
                title='Revenue Box Plot by Platform',
                color_discrete_map=PLATFORM_COLORS
            )
            fig_box.update_layout(**create_enhanced_visualizations())
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True, key='revenue_boxplot')

    with tab3:
        st.markdown("### 🏆 Performance Analytics")
        
        # Store performance
        if 'Store_Name' in completed_df.columns:
            store_performance = completed_df.groupby('Store_Name')['Revenue'].agg(['sum', 'count', 'mean']).round(2)
            store_performance.columns = ['Total_Revenue', 'Order_Count', 'Avg_Order_Value']
            store_performance = store_performance.sort_values('Total_Revenue', ascending=False)
            
            # Top performing stores
            fig_stores = px.bar(
                store_performance.head(10).reset_index(),
                x='Store_Name',
                y='Total_Revenue',
                title='Top 10 Stores by Revenue',
                color='Total_Revenue',
                color_continuous_scale='Oranges'
            )
            fig_stores.update_layout(**create_enhanced_visualizations())
            fig_stores.update_layout(xaxis_tickangle=-45, yaxis_title="Total Revenue ($)")
            st.plotly_chart(fig_stores, use_container_width=True, key='performance_top_stores')
            
            # Store performance table
            st.markdown("### 🏪 Store Performance Details")
            display_stores = store_performance.copy()
            display_stores['Total_Revenue'] = display_stores['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
            display_stores['Avg_Order_Value'] = display_stores['Avg_Order_Value'].apply(lambda x: f"${x:.2f}")
            st.dataframe(display_stores, use_container_width=True)

    with tab4:
        st.markdown("### 🕐 Operational Analytics")
        
        # Hourly analysis
        if 'Hour' in completed_df.columns and completed_df['Hour'].notna().any():
            hourly_orders = completed_df.groupby(['Hour', 'Platform']).size().reset_index(name='Orders')
            
            fig_hourly = px.bar(
                hourly_orders,
                x='Hour',
                y='Orders',
                color='Platform',
                title='Orders by Hour of Day',
                color_discrete_map=PLATFORM_COLORS,
                barmode='group'
            )
            fig_hourly.update_layout(**create_enhanced_visualizations())
            fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Number of Orders")
            st.plotly_chart(fig_hourly, use_container_width=True, key='operations_hourly')
        
        # Day of week analysis
        if 'DayOfWeek' in completed_df.columns and completed_df['DayOfWeek'].notna().any():
            dow_orders = completed_df.groupby(['DayOfWeek', 'Platform']).agg({
                'Revenue': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            dow_orders.columns = ['DayOfWeek', 'Platform', 'Revenue', 'Orders']
            
            # Reorder days of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_orders['DayOfWeek'] = pd.Categorical(dow_orders['DayOfWeek'], categories=day_order, ordered=True)
            dow_orders = dow_orders.sort_values('DayOfWeek')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dow_orders = px.bar(
                    dow_orders,
                    x='DayOfWeek',
                    y='Orders',
                    color='Platform',
                    title='Orders by Day of Week',
                    color_discrete_map=PLATFORM_COLORS,
                    barmode='group'
                )
                fig_dow_orders.update_layout(**create_enhanced_visualizations())
                fig_dow_orders.update_layout(xaxis_title="Day of Week", xaxis_tickangle=-45)
                st.plotly_chart(fig_dow_orders, use_container_width=True, key='operations_dow_orders')
            
            with col2:
                fig_dow_revenue = px.bar(
                    dow_orders,
                    x='DayOfWeek',
                    y='Revenue',
                    color='Platform',
                    title='Revenue by Day of Week',
                    color_discrete_map=PLATFORM_COLORS,
                    barmode='group'
                )
                fig_dow_revenue.update_layout(**create_enhanced_visualizations())
                fig_dow_revenue.update_layout(xaxis_title="Day of Week", xaxis_tickangle=-45, yaxis_title="Revenue ($)")
                st.plotly_chart(fig_dow_revenue, use_container_width=True, key='operations_dow_revenue')

    with tab5:
        st.markdown("### 📈 Growth & Trends Analysis")
        
        # Monthly trends
        if 'Month' in completed_df.columns:
            monthly_trends = completed_df.groupby(['Month', 'Platform']).agg({
                'Revenue': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            monthly_trends.columns = ['Month', 'Platform', 'Revenue', 'Orders']
            monthly_trends['Month_Str'] = monthly_trends['Month'].astype(str)
            
            # Revenue growth trend
            fig_monthly_revenue = px.line(
                monthly_trends,
                x='Month_Str',
                y='Revenue',
                color='Platform',
                title='Monthly Revenue Growth by Platform',
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            fig_monthly_revenue.update_layout(**create_enhanced_visualizations())
            fig_monthly_revenue.update_layout(xaxis_title="Month", yaxis_title="Monthly Revenue ($)")
            st.plotly_chart(fig_monthly_revenue, use_container_width=True, key='trends_monthly_revenue')
            
            # Orders growth trend
            fig_monthly_orders = px.line(
                monthly_trends,
                x='Month_Str',
                y='Orders',
                color='Platform',
                title='Monthly Orders Growth by Platform',
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            fig_monthly_orders.update_layout(**create_enhanced_visualizations())
            fig_monthly_orders.update_layout(xaxis_title="Month", yaxis_title="Monthly Orders")
            st.plotly_chart(fig_monthly_orders, use_container_width=True, key='trends_monthly_orders')

    with tab6:
        st.markdown("### 👥 Customer Attribution Analysis")
        
        # Customer segments based on order frequency and revenue
        if 'Order_ID' in completed_df.columns:
            customer_analysis = completed_df.groupby('Order_ID').agg({
                'Revenue': ['sum', 'count'],
                'Date': ['min', 'max']
            }).round(2)
            
            customer_analysis.columns = ['Total_Revenue', 'Order_Frequency', 'First_Order', 'Last_Order']
            customer_analysis['Days_Active'] = (customer_analysis['Last_Order'] - customer_analysis['First_Order']).dt.days + 1
            
            # Create customer segments
            revenue_quartiles = customer_analysis['Total_Revenue'].quantile([0.25, 0.5, 0.75])
            frequency_quartiles = customer_analysis['Order_Frequency'].quantile([0.25, 0.5, 0.75])
            
            def classify_customer(row):
                if row['Total_Revenue'] >= revenue_quartiles[0.75] and row['Order_Frequency'] >= frequency_quartiles[0.75]:
                    return 'VIP'
                elif row['Total_Revenue'] >= revenue_quartiles[0.5] and row['Order_Frequency'] >= frequency_quartiles[0.5]:
                    return 'Loyal'
                elif row['Total_Revenue'] >= revenue_quartiles[0.25]:
                    return 'Regular'
                else:
                    return 'Casual'
            
            customer_analysis['Segment'] = customer_analysis.apply(classify_customer, axis=1)
            
            # Customer segments distribution
            segment_summary = customer_analysis.groupby('Segment').agg({
                'Total_Revenue': ['count', 'sum', 'mean'],
                'Order_Frequency': 'mean'
            }).round(2)
            
            segment_summary.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Revenue', 'Avg_Frequency']
            segment_summary = segment_summary.reset_index()
            
            # Visualize customer segments
            fig_segments = px.bar(
                segment_summary,
                x='Segment',
                y='Customer_Count',
                color='Segment',
                title='Customer Distribution by Segment',
                color_discrete_sequence=['#8B4513', '#D2691E', '#CD853F', '#F4A460']
            )
            fig_segments.update_layout(**create_enhanced_visualizations())
            fig_segments.update_layout(showlegend=False, yaxis_title="Number of Customers")
            st.plotly_chart(fig_segments, use_container_width=True, key='attribution_segments')
            
            # Segment analysis table
            st.markdown("### 📊 Customer Segment Analysis")
            display_segments = segment_summary.copy()
            display_segments['Total_Revenue'] = display_segments['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
            display_segments['Avg_Revenue'] = display_segments['Avg_Revenue'].apply(lambda x: f"${x:.2f}")
            st.dataframe(display_segments, use_container_width=True, hide_index=True)

    with tab7:
        st.markdown("### 🔄 Retention & Churn Analysis")
        
        # Calculate retention metrics
        if 'Order_ID' in completed_df.columns and 'Date' in completed_df.columns:
            # Customer ordering pattern
            customer_orders = completed_df.groupby('Order_ID')['Date'].agg(['min', 'max', 'count']).reset_index()
            customer_orders.columns = ['Order_ID', 'First_Order', 'Last_Order', 'Total_Orders']
            customer_orders['Days_Since_First'] = (customer_orders['Last_Order'] - customer_orders['First_Order']).dt.days
            customer_orders['Days_Since_Last'] = (datetime.now().date() - customer_orders['Last_Order'].dt.date).dt.days
            
            # Define churn threshold
            churn_threshold = st.selectbox("Churn Threshold (days since last order)", [30, 60, 90], index=1)
            customer_orders['Is_Churned'] = customer_orders['Days_Since_Last'] > churn_threshold
            
            # Retention rate
            total_customers = len(customer_orders)
            active_customers = len(customer_orders[~customer_orders['Is_Churned']])
            retention_rate = (active_customers / total_customers * 100) if total_customers > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #8B4513; margin: 0;">Total Customers</h3>
                        <h2 style="color: #D2691E; margin: 0.5rem 0;">{total_customers:,}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #8B4513; margin: 0;">Active Customers</h3>
                        <h2 style="color: #D2691E; margin: 0.5rem 0;">{active_customers:,}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #8B4513; margin: 0;">Retention Rate</h3>
                        <h2 style="color: #D2691E; margin: 0.5rem 0;">{retention_rate:.1f}%</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # Order frequency analysis
            frequency_dist = customer_orders['Total_Orders'].value_counts().sort_index()
            
            fig_frequency = px.bar(
                x=frequency_dist.index,
                y=frequency_dist.values,
                title='Customer Order Frequency Distribution',
                color=frequency_dist.values,
                color_continuous_scale='Oranges'
            )
            fig_frequency.update_layout(**create_enhanced_visualizations())
            fig_frequency.update_layout(xaxis_title="Number of Orders", yaxis_title="Number of Customers")
            st.plotly_chart(fig_frequency, use_container_width=True, key='retention_frequency')

    with tab8:
        st.markdown("### 📱 Comprehensive Platform Comparison")
        
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
            if 'Hour' in platform_data.columns and platform_data['Hour'].notna().any():
                metrics_dict['Peak Hour'] = platform_data.groupby('Hour').size().idxmax() if len(platform_data) > 0 else 0
            
            # Add top day if available
            if 'DayOfWeek' in platform_data.columns and platform_data['DayOfWeek'].notna().any():
                metrics_dict['Top Day'] = platform_data.groupby('DayOfWeek').size().idxmax() if len(platform_data) > 0 else 'N/A'
            
            comparison_metrics = pd.concat([comparison_metrics, pd.DataFrame([metrics_dict])], ignore_index=True)
        
        # Display comparison table
        st.markdown("### 📊 Key Performance Indicators")
        
        # Format the metrics for display
        if not comparison_metrics.empty:
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
                    line_color=PLATFORM_COLORS.get(row['Platform'], '#8B4513')
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                title="Platform Performance Radar Chart (Normalized)",
                **create_enhanced_visualizations()
            )
            
            st.plotly_chart(fig_radar, use_container_width=True, key='comparison_radar_chart')

    # Export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Analytics Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report", key="export_excel_button"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                platform_summary.to_excel(writer, sheet_name='Platform_Summary', index=False)
                
                # Revenue analysis
                daily_revenue.to_excel(writer, sheet_name='Daily_Revenue', index=False)
                
                # Processed data
                completed_df.to_excel(writer, sheet_name='All_Data', index=False)
                
                # Platform comparison
                if not comparison_metrics.empty:
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
            platform_revenue = completed_df.groupby('Platform')['Revenue'].sum()
            platform_orders = completed_df.groupby('Platform').size()
            
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
{platform_summary.to_string()}

TOP INSIGHTS
============
1. Highest revenue platform: {platform_revenue.idxmax() if not platform_revenue.empty else 'N/A'}
2. Most orders platform: {platform_orders.idxmax() if not platform_orders.empty else 'N/A'}
3. Data quality: Successfully processed {len(completed_df)} valid orders

Date Range: {completed_df['Date'].min().strftime('%Y-%m-%d')} to {completed_df['Date'].max().strftime('%Y-%m-%d')}
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
        <div style='text-align: center; color: #6c757d; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;'>
            <p style="font-size: 1.2rem; margin-bottom: 0.5rem; font-family: 'Crimson Pro', serif; color: #8B4513;">☕ Luckin Coffee Marketing Analytics Dashboard v4.0</p>
            <p style='font-size: 0.9rem; margin: 0; color: #6c757d;'>Enhanced Error Handling | Multi-Platform Data Processing | Coffee-Themed Design</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
