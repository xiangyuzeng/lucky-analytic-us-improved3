#!/usr/bin/env python3
"""
CSV File Validator for Luckin Coffee Analytics Dashboard
This script helps diagnose issues with CSV files before running the main application.
"""

import pandas as pd
import sys
import os

def validate_doordash(file_path):
    """Validate DoorDash CSV file"""
    print("\n📊 Validating DoorDash CSV...")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ File loaded successfully: {len(df)} rows")
        
        required_cols = ['时间戳本地日期', '净总计']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns[:10])}")
            return False
        
        print("✅ All required columns present")
        
        # Check data types
        dates = pd.to_datetime(df['时间戳本地日期'], errors='coerce')
        valid_dates = dates.notna().sum()
        print(f"📅 Valid dates: {valid_dates}/{len(df)}")
        
        revenue = pd.to_numeric(df['净总计'], errors='coerce')
        valid_revenue = revenue.notna().sum()
        print(f"💰 Valid revenue values: {valid_revenue}/{len(df)}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_uber(file_path):
    """Validate Uber CSV file"""
    print("\n📊 Validating Uber CSV...")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ File loaded successfully: {len(df)} rows")
        
        # Check if first row contains descriptions
        first_row = df.iloc[0] if len(df) > 0 else None
        
        if first_row is not None and (
            'Uber Eats 优食管理工具中显示的餐厅名称' in df.columns[0] or
            first_row.astype(str).str.contains('优食管理工具|识别编号').any()
        ):
            print("📝 Detected description row, adjusting headers...")
            # Use the second row as headers
            if len(df) > 1:
                df.columns = df.iloc[1].astype(str).str.strip()
                df = df.iloc[2:].reset_index(drop=True)
                print(f"✅ Adjusted to {len(df)} data rows")
        
        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Check for required columns
        required_cols = ['订单日期', '收入总额']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ Missing expected columns: {missing_cols}")
            print(f"Available columns (first 10): {list(df.columns[:10])}")
        else:
            print("✅ All required columns present")
            
            # Check data types
            dates = pd.to_datetime(df['订单日期'], errors='coerce')
            valid_dates = dates.notna().sum()
            print(f"📅 Valid dates: {valid_dates}/{len(df)}")
            
            revenue = pd.to_numeric(df['收入总额'], errors='coerce')
            valid_revenue = revenue.notna().sum()
            print(f"💰 Valid revenue values: {valid_revenue}/{len(df)}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_grubhub(file_path):
    """Validate Grubhub CSV file"""
    print("\n📊 Validating Grubhub CSV...")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ File loaded successfully: {len(df)} rows")
        
        # Check for required columns
        revenue_cols = ['merchant_net_total', 'merchant_total']
        has_revenue = any(col in df.columns for col in revenue_cols)
        
        if not has_revenue:
            print(f"❌ Missing revenue columns. Need one of: {revenue_cols}")
            print(f"Available columns: {list(df.columns[:10])}")
            return False
        
        print("✅ Revenue column found")
        
        # Check dates
        if 'transaction_date' in df.columns:
            date_series = df['transaction_date'].astype(str)
            if date_series.str.contains(r'E\+|#', regex=True).any():
                print("⚠️ Warning: Dates appear corrupted (Excel format issue)")
                print("   The application will use fallback date handling")
            else:
                dates = pd.to_datetime(date_series, errors='coerce')
                valid_dates = dates.notna().sum()
                print(f"📅 Valid dates: {valid_dates}/{len(df)}")
        else:
            print("⚠️ No date column found, will use current date")
        
        # Check revenue
        revenue_col = 'merchant_net_total' if 'merchant_net_total' in df.columns else 'merchant_total'
        revenue = pd.to_numeric(df[revenue_col], errors='coerce')
        valid_revenue = revenue.notna().sum()
        print(f"💰 Valid revenue values: {valid_revenue}/{len(df)}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Luckin Coffee CSV File Validator")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python validate_csv.py <csv_file> [platform]")
        print("Platform options: doordash, uber, grubhub")
        print("\nExample: python validate_csv.py data.csv doordash")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Try to determine platform from filename or argument
    platform = None
    if len(sys.argv) >= 3:
        platform = sys.argv[2].lower()
    else:
        # Try to guess from filename
        filename_lower = file_path.lower()
        if 'doordash' in filename_lower:
            platform = 'doordash'
        elif 'uber' in filename_lower:
            platform = 'uber'
        elif 'grubhub' in filename_lower:
            platform = 'grubhub'
    
    if platform == 'doordash':
        validate_doordash(file_path)
    elif platform == 'uber':
        validate_uber(file_path)
    elif platform == 'grubhub':
        validate_grubhub(file_path)
    else:
        print("\n🔍 Auto-detecting platform...")
        # Try to detect based on column names
        try:
            df = pd.read_csv(file_path)
            cols = df.columns.tolist()
            
            if '时间戳本地日期' in cols or 'DoorDash 订单 ID' in cols:
                print("Detected: DoorDash")
                validate_doordash(file_path)
            elif '订单日期' in cols or '餐厅名称' in cols:
                print("Detected: Uber")
                validate_uber(file_path)
            elif 'merchant_net_total' in cols or 'order_number' in cols:
                print("Detected: Grubhub")
                validate_grubhub(file_path)
            else:
                print("❌ Could not detect platform. Please specify: doordash, uber, or grubhub")
                print(f"First few columns: {cols[:5]}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
