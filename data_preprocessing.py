import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = []
        
    def generate_synthetic_bearing_data(self, n_samples=10000):
        """Generate synthetic bearing data similar to NASA dataset"""
        np.random.seed(42)
        
        # Time vector
        time = np.linspace(0, 100, n_samples)
        
        # Simulate bearing degradation over time
        degradation_factor = np.exp(time / 50)  # Exponential degradation
        
        # Base vibration signals with increasing amplitude due to wear
        vibration_x = np.sin(2 * np.pi * 10 * time) * degradation_factor + np.random.normal(0, 0.1, n_samples)
        vibration_y = np.cos(2 * np.pi * 15 * time) * degradation_factor + np.random.normal(0, 0.1, n_samples)
        
        # Add bearing fault frequencies
        fault_freq = 0.5 * time  # Increasing fault frequency
        vibration_x += 0.3 * np.sin(2 * np.pi * fault_freq) * (degradation_factor > 2)
        vibration_y += 0.3 * np.cos(2 * np.pi * fault_freq) * (degradation_factor > 2)
        
        # Temperature increases with wear
        temperature = 25 + 0.1 * time + np.random.normal(0, 1, n_samples)
        
        # RPM with slight variations
        rpm = 1800 + 50 * np.sin(2 * np.pi * 0.1 * time) + np.random.normal(0, 10, n_samples)
        
        # Health labels (0: Healthy, 1: Faulty)
        health_labels = (degradation_factor > 2.5).astype(int)
        
        # RUL (Remaining Useful Life) - decreases over time
        max_life = 100
        rul = np.maximum(0, max_life - time)
        
        data = pd.DataFrame({
            'time': time,
            'vibration_x': vibration_x,
            'vibration_y': vibration_y,
            'temperature': temperature,
            'rpm': rpm,
            'health_label': health_labels,
            'rul': rul
        })
        
        return data
    
    def extract_time_domain_features(self, signal):
        """Extract time-domain features from vibration signal"""
        # Convert pandas Series to numpy array if needed
        if hasattr(signal, 'values'):
            signal = signal.values
        signal = np.array(signal)
        
        features = {}
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['variance'] = np.var(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['skewness'] = stats.skew(signal)
        features['peak'] = np.max(np.abs(signal))
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        
        return features
    
    def extract_frequency_domain_features(self, signal, fs=1000):
        """Extract frequency-domain features using FFT"""
        # Convert pandas Series to numpy array if needed
        if hasattr(signal, 'values'):
            signal = signal.values
        signal = np.array(signal)
        
        # Compute FFT
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), 1/fs)
        
        # Take only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_magnitude = np.abs(fft_vals[pos_mask])
        
        features = {}
        total_magnitude = np.sum(fft_magnitude)
        if total_magnitude > 0:
            features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / total_magnitude
            features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * fft_magnitude) / total_magnitude)
            
            # Find spectral rolloff
            cumsum_magnitude = np.cumsum(fft_magnitude)
            rolloff_idx = np.where(cumsum_magnitude >= 0.85 * total_magnitude)[0]
            features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            features['spectral_centroid'] = 0
            features['spectral_bandwidth'] = 0
            features['spectral_rolloff'] = 0
            
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        features['dominant_freq'] = freqs[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0
        
        return features
    
    def create_feature_matrix(self, data, window_size=100):
        """Create feature matrix from raw sensor data"""
        features_list = []
        labels_list = []
        rul_list = []
        
        for i in range(0, len(data) - window_size, window_size//2):
            window = data.iloc[i:i+window_size]
            
            # Extract features for each signal
            vib_x_features = self.extract_time_domain_features(window['vibration_x'])
            vib_y_features = self.extract_time_domain_features(window['vibration_y'])
            
            vib_x_freq_features = self.extract_frequency_domain_features(window['vibration_x'])
            vib_y_freq_features = self.extract_frequency_domain_features(window['vibration_y'])
            
            # Combine all features
            combined_features = {}
            for key, value in vib_x_features.items():
                combined_features[f'vib_x_{key}'] = value
            for key, value in vib_y_features.items():
                combined_features[f'vib_y_{key}'] = value
            for key, value in vib_x_freq_features.items():
                combined_features[f'vib_x_freq_{key}'] = value
            for key, value in vib_y_freq_features.items():
                combined_features[f'vib_y_freq_{key}'] = value
            
            # Add other sensor features
            combined_features['temp_mean'] = window['temperature'].mean()
            combined_features['temp_std'] = window['temperature'].std()
            combined_features['rpm_mean'] = window['rpm'].mean()
            combined_features['rpm_std'] = window['rpm'].std()
            
            features_list.append(combined_features)
            labels_list.append(window['health_label'].iloc[-1])  # Use last label in window
            rul_list.append(window['rul'].iloc[-1])  # Use last RUL in window
        
        feature_df = pd.DataFrame(features_list)
        feature_df['health_label'] = labels_list
        feature_df['rul'] = rul_list
        
        self.features = [col for col in feature_df.columns if col not in ['health_label', 'rul']]
        
        return feature_df
    
    def preprocess_data(self, data):
        """Complete preprocessing pipeline"""
        # Create feature matrix
        feature_data = self.create_feature_matrix(data)
        
        # Separate features and targets
        X = feature_data[self.features]
        y_classification = feature_data['health_label']
        y_regression = feature_data['rul']
        
        # Handle any NaN values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
            X, y_classification, y_regression, test_size=0.4, random_state=42, stratify=y_classification
        )
        
        X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
            X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42, stratify=y_class_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_class_train': y_class_train,
            'y_class_val': y_class_val,
            'y_class_test': y_class_test,
            'y_reg_train': y_reg_train,
            'y_reg_val': y_reg_val,
            'y_reg_test': y_reg_test,
            'feature_names': self.features,
            'raw_features': X
        }

def load_and_preprocess_data():
    """Main function to load and preprocess data"""
    preprocessor = DataPreprocessor()
    
    # Generate synthetic data (in real scenario, load from NASA dataset)
    raw_data = preprocessor.generate_synthetic_bearing_data()
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data(raw_data)
    
    return processed_data, raw_data, preprocessor