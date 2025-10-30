#!/usr/bin/env python3
"""
Test script to verify all components work before running Streamlit
"""

def test_data_preprocessing():
    print("Testing data preprocessing...")
    try:
        from data_preprocessing import load_and_preprocess_data
        processed_data, raw_data, preprocessor = load_and_preprocess_data()
        print(f"✅ Data preprocessing successful!")
        print(f"   - Raw data shape: {raw_data.shape}")
        print(f"   - Training data shape: {processed_data['X_train'].shape}")
        print(f"   - Features extracted: {len(preprocessor.features)}")
        return processed_data, raw_data, preprocessor
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None, None, None

def test_model_training(processed_data):
    print("\nTesting model training...")
    try:
        from model_training import train_all_models
        trainer, results, stats = train_all_models(processed_data)
        print(f"✅ Model training successful!")
        print(f"   - Models trained: {len(trainer.models)}")
        print(f"   - Results: {list(results.keys())}")
        return trainer, results, stats
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None, None, None

def test_inference_engine():
    print("\nTesting inference engine...")
    try:
        from inference_tflite import create_inference_engine
        engine = create_inference_engine()
        info = engine.get_model_info()
        print(f"✅ Inference engine successful!")
        print(f"   - Models loaded: {info['total_models']}")
        print(f"   - Model names: {info['loaded_models']}")
        return engine
    except Exception as e:
        print(f"❌ Inference engine failed: {e}")
        return None

def main():
    print("🚀 Testing TinyML Predictive Maintenance Dashboard Components\n")
    
    # Test data preprocessing
    processed_data, raw_data, preprocessor = test_data_preprocessing()
    if processed_data is None:
        return
    
    # Test model training
    trainer, results, stats = test_model_training(processed_data)
    if trainer is None:
        return
    
    # Test inference engine
    engine = test_inference_engine()
    if engine is None:
        return
    
    # Test a sample prediction
    print("\nTesting sample prediction...")
    try:
        sample_data = processed_data['X_test'][0]
        results = engine.predict_all_models(sample_data)
        print(f"✅ Sample prediction successful!")
        print(f"   - Predictions from {len(results)} models")
        for model_name, result in results.items():
            pred = result['prediction'][0] if hasattr(result['prediction'], '__len__') else result['prediction']
            print(f"   - {model_name}: {pred:.3f} ({result['inference_time_ms']:.2f}ms)")
    except Exception as e:
        print(f"❌ Sample prediction failed: {e}")
        return
    
    print(f"\n🎉 All tests passed! The dashboard should work correctly.")
    print(f"Run: python -m streamlit run app.py")

if __name__ == "__main__":
    main()