import numpy as np
import time
import os
import joblib

class InferenceEngine:
    def __init__(self):
        self.models = {}
        
    def load_all_models(self):
        """Load all available models"""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            print("Models directory not found. Please train models first.")
            return
        
        # Load all pickle models
        pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        for pkl_file in pkl_files:
            model_name = pkl_file.replace('.pkl', '')
            model_path = os.path.join(models_dir, pkl_file)
            try:
                model = joblib.load(model_path)
                self.models[model_name] = model
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
    
    def predict_model(self, model, input_data):
        """Perform inference using any sklearn model"""
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        start_time = time.time()
        
        # For classification models, get probabilities if available
        if hasattr(model, 'predict_proba') and 'classifier' in str(type(model)).lower():
            prediction = model.predict_proba(input_data)[:, 1]  # Get probability of positive class
        else:
            prediction = model.predict(input_data)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return prediction, inference_time
    
    def predict_all_models(self, input_data):
        """Run inference on all loaded models"""
        results = {}
        
        # All models
        for model_name, model in self.models.items():
            try:
                prediction, inference_time = self.predict_model(model, input_data)
                model_type = 'TinyML-style' if 'mlp' in model_name else 'Traditional ML'
                results[model_name] = {
                    'prediction': prediction,
                    'inference_time_ms': inference_time,
                    'model_type': model_type
                }
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.models.keys()),
            'total_models': len(self.models)
        }
        
        # Get model sizes
        models_dir = 'models'
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                filepath = os.path.join(models_dir, filename)
                size_kb = os.path.getsize(filepath) / 1024
                info[f'{filename}_size_kb'] = round(size_kb, 2)
        
        return info
    
    def benchmark_models(self, input_data, num_runs=100):
        """Benchmark inference speed of all models"""
        benchmark_results = {}
        
        print(f"Benchmarking models with {num_runs} runs...")
        
        # Benchmark all models
        for model_name, model in self.models.items():
            times = []
            for _ in range(num_runs):
                _, inference_time = self.predict_model(model, input_data)
                times.append(inference_time)
            
            model_type = 'TinyML-style' if 'mlp' in model_name else 'Traditional ML'
            benchmark_results[model_name] = {
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'model_type': model_type
            }
        
        return benchmark_results

def create_inference_engine():
    """Create and initialize inference engine"""
    engine = InferenceEngine()
    engine.load_all_models()
    return engine