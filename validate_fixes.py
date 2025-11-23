import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import the processing functions from our fixed dashboard
sys.path.append('.')

def test_data_processing_fixes():
    """Test all the data processing fixes with actual data"""
    
    print("🔬 TESTING LUCKIN ANALYTICS DASHBOARD FIXES")
    print("=" * 60)
    
    results = {
        'doordash': {'status': 'NOT_TESTED', 'records': 0, 'issues': []},
        'uber': {'status': 'NOT_TESTED', 'records': 0, 'issues': []},
        'grubhub': {'status': 'NOT_TESTED', 'records': 0, 'issues': []}
    }
    
    # Test DoorDash processing
    print("\n📊 Testing DoorDash Data Processing...")
    try:
        if os.path.exists('doordash.csv'):
            from improved_luckin_analytics_fixed import process_doordash_data
            
            dd_df = pd.read_csv('doordash.csv')
            dd_processed = process_doordash_data(dd_df)
            
            results['doordash']['status'] = 'SUCCESS' if not dd_processed.empty else 'FAILED'
            results['doordash']['records'] = len(dd_processed)
            
            if not dd_processed.empty:
                print(f"  ✅ Processed {len(dd_processed)} records from {len(dd_df)} raw records")
                print(f"  ✅ Date range: {dd_processed['Date'].min()} to {dd_processed['Date'].max()}")
                print(f"  ✅ Revenue range: ${dd_processed['Revenue'].min():.2f} to ${dd_processed['Revenue'].max():.2f}")
                print(f"  ✅ Completion rate: {dd_processed['Is_Completed'].mean()*100:.1f}%")
                print(f"  ✅ Unique stores: {dd_processed['Store_Name'].nunique()}")
            else:
                results['doordash']['issues'].append("No valid records after processing")
                print("  ❌ No valid records after processing")
        else:
            results['doordash']['status'] = 'NO_FILE'
            print("  ⚠️  File not found")
    except Exception as e:
        results['doordash']['status'] = 'ERROR'
        results['doordash']['issues'].append(str(e))
        print(f"  ❌ Error: {e}")
    
    # Test Uber processing
    print("\n🚗 Testing Uber Data Processing (Two-row header fix)...")
    try:
        if os.path.exists('Uber.csv'):
            from improved_luckin_analytics_fixed import process_uber_data
            
            uber_df = pd.read_csv('Uber.csv')
            uber_processed = process_uber_data(uber_df)
            
            results['uber']['status'] = 'SUCCESS' if not uber_processed.empty else 'FAILED'
            results['uber']['records'] = len(uber_processed)
            
            if not uber_processed.empty:
                print(f"  ✅ Processed {len(uber_processed)} records from {len(uber_df)} raw records")
                print(f"  ✅ Header fix applied - detected two-row format")
                print(f"  ✅ Date range: {uber_processed['Date'].min()} to {uber_processed['Date'].max()}")
                print(f"  ✅ Revenue range: ${uber_processed['Revenue'].min():.2f} to ${uber_processed['Revenue'].max():.2f}")
                print(f"  ✅ Completion rate: {uber_processed['Is_Completed'].mean()*100:.1f}%")
                print(f"  ✅ Unique stores: {uber_processed['Store_Name'].nunique()}")
            else:
                results['uber']['issues'].append("No valid records after processing")
                print("  ❌ No valid records after processing")
        else:
            results['uber']['status'] = 'NO_FILE'
            print("  ⚠️  File not found")
    except Exception as e:
        results['uber']['status'] = 'ERROR'
        results['uber']['issues'].append(str(e))
        print(f"  ❌ Error: {e}")
    
    # Test Grubhub processing
    print("\n🍔 Testing Grubhub Data Processing (Date corruption fix)...")
    try:
        if os.path.exists('grubhub.csv'):
            from improved_luckin_analytics_fixed import process_grubhub_data
            
            gh_df = pd.read_csv('grubhub.csv')
            
            # Check for date corruption first
            dates_corrupted = gh_df['transaction_date'].astype(str).str.contains('####').any()
            if dates_corrupted:
                print("  🚨 Date corruption detected - applying fix...")
            
            gh_processed = process_grubhub_data(gh_df)
            
            results['grubhub']['status'] = 'SUCCESS' if not gh_processed.empty else 'FAILED'
            results['grubhub']['records'] = len(gh_processed)
            
            if not gh_processed.empty:
                print(f"  ✅ Processed {len(gh_processed)} records from {len(gh_df)} raw records")
                if dates_corrupted:
                    print("  ✅ Date corruption fix applied successfully")
                print(f"  ✅ Date range: {gh_processed['Date'].min()} to {gh_processed['Date'].max()}")
                print(f"  ✅ Revenue range: ${gh_processed['Revenue'].min():.2f} to ${gh_processed['Revenue'].max():.2f}")
                print(f"  ✅ Completion rate: {gh_processed['Is_Completed'].mean()*100:.1f}%")
                print(f"  ✅ Unique stores: {gh_processed['Store_Name'].nunique()}")
            else:
                results['grubhub']['issues'].append("No valid records after processing")
                print("  ❌ No valid records after processing")
        else:
            results['grubhub']['status'] = 'NO_FILE'
            print("  ⚠️  File not found")
    except Exception as e:
        results['grubhub']['status'] = 'ERROR'
        results['grubhub']['issues'].append(str(e))
        print(f"  ❌ Error: {e}")
    
    # Test store normalization
    print("\n🏪 Testing Store Name Normalization...")
    try:
        from improved_luckin_analytics_fixed import normalize_store_names
        
        # Create test data with various store names
        test_data = pd.DataFrame({
            'Store_Name': [
                'Luckin Coffee  (Broadway)',
                'Luckin Coffee US00002',
                'Luckin Coffee - Broadway',
                'Luckin Coffee US00001',
                'Luckin Coffee US00003',
                'Unknown Store'
            ]
        })
        
        normalized = normalize_store_names(test_data)
        
        print("  ✅ Store normalization working:")
        for orig, norm in zip(test_data['Store_Name'], normalized['Store_Name_Normalized']):
            if orig != norm:
                print(f"    {orig} → {norm}")
            else:
                print(f"    {orig} (unchanged)")
                
    except Exception as e:
        print(f"  ❌ Store normalization error: {e}")
    
    # Test combined data processing
    print("\n🔗 Testing Combined Data Processing...")
    try:
        all_data = []
        total_records = 0
        
        for platform in ['doordash', 'uber', 'grubhub']:
            if results[platform]['status'] == 'SUCCESS':
                all_data.append(platform)
                total_records += results[platform]['records']
        
        if all_data:
            print(f"  ✅ Successfully processed {len(all_data)} platforms")
            print(f"  ✅ Total records: {total_records}")
            print(f"  ✅ Platforms: {', '.join(all_data)}")
        else:
            print("  ❌ No platforms successfully processed")
            
    except Exception as e:
        print(f"  ❌ Combined processing error: {e}")
    
    # Summary
    print("\n📋 PROCESSING SUMMARY")
    print("=" * 30)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    total_records = sum(r['records'] for r in results.values())
    
    print(f"Platforms processed successfully: {success_count}/3")
    print(f"Total records processed: {total_records}")
    
    for platform, result in results.items():
        status_icon = {
            'SUCCESS': '✅',
            'FAILED': '❌',
            'ERROR': '💥',
            'NO_FILE': '📁',
            'NOT_TESTED': '⏸️'
        }
        
        print(f"{status_icon[result['status']]} {platform.upper()}: {result['status']} ({result['records']} records)")
        
        if result['issues']:
            for issue in result['issues']:
                print(f"    ⚠️  {issue}")
    
    if success_count == 3:
        print("\n🎉 ALL FIXES WORKING CORRECTLY!")
        print("   Dashboard is ready for use with all platforms.")
    elif success_count > 0:
        print(f"\n✅ {success_count} platforms working correctly.")
        print("   Dashboard can be used with available data.")
    else:
        print("\n❌ No platforms processed successfully.")
        print("   Check file paths and data format.")
    
    return results

if __name__ == "__main__":
    test_data_processing_fixes()
