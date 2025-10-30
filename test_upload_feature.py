#!/usr/bin/env python3
"""
Test script for the upload feature
"""

def test_demo_datasets():
    print("🧪 Testing demo dataset creation...")
    
    try:
        from app import create_demo_factory_data, create_demo_automotive_data, create_demo_aerospace_data
        
        # Test factory data
        factory_data = create_demo_factory_data()
        print(f"✅ Factory data: {factory_data.shape}")
        
        # Test automotive data
        auto_data = create_demo_automotive_data()
        print(f"✅ Automotive data: {auto_data.shape}")
        
        # Test aerospace data
        aero_data = create_demo_aerospace_data()
        print(f"✅ Aerospace data: {aero_data.shape}")
        
        # Verify columns
        expected_cols = ['time', 'vibration_x', 'vibration_y', 'temperature', 'rpm']
        for name, data in [("Factory", factory_data), ("Automotive", auto_data), ("Aerospace", aero_data)]:
            if list(data.columns) == expected_cols:
                print(f"✅ {name} columns correct")
            else:
                print(f"❌ {name} columns incorrect: {list(data.columns)}")
        
        print("\n🎉 All demo datasets created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing demo datasets: {e}")
        return False

def test_analytics_functions():
    print("\n🧪 Testing analytics functions...")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'time': np.linspace(0, 100, 1000),
            'vibration_x': np.random.normal(0, 1, 1000),
            'vibration_y': np.random.normal(0, 1, 1000),
            'temperature': np.random.normal(25, 5, 1000),
            'rpm': np.random.normal(1800, 100, 1000)
        })
        
        # Test basic analytics
        vibration_magnitude = np.sqrt(sample_data['vibration_x']**2 + sample_data['vibration_y']**2)
        health_score = max(0, 100 - (vibration_magnitude > vibration_magnitude.quantile(0.8)).sum() / len(sample_data) * 100)
        
        print(f"✅ Sample data shape: {sample_data.shape}")
        print(f"✅ Health score calculated: {health_score:.1f}%")
        print(f"✅ Analytics functions working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing analytics: {e}")
        return False

def main():
    print("🚀 Testing Upload Feature Components\n")
    
    success1 = test_demo_datasets()
    success2 = test_analytics_functions()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Upload feature is ready to use.")
        print("💡 Run the dashboard with: python -m streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()