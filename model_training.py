import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report
import joblib
import os
import pickle

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.history = {}
        
    def create_mlp_model(self, input_shape, task='classification'):
        """Create MLP model using scikit-learn (TinyML-style)"""
        if task == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=(32, 16, 8),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
        else:  # regression
            model = MLPRegressor(
                hidden_layer_sizes=(32, 16, 8),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
        return model
    
    def train_baseline_models(self, X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val):
        """Train baseline ML models"""
        print("Training baseline models...")
        
        # Random Forest for classification
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_class_train)
        
        # Random Forest for regression
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_reg_train)
        
        # SVM for classification
        svm_classifier = SVC(kernel='rbf', random_state=42, probability=True)
        svm_classifier.fit(X_train, y_class_train)
        
        # SVM for regression
        svm_regressor = SVR(kernel='rbf')
        svm_regressor.fit(X_train, y_reg_train)
        
        # Store models
        self.models['rf_classifier'] = rf_classifier
        self.models['rf_regressor'] = rf_regressor
        self.models['svm_classifier'] = svm_classifier
        self.models['svm_regressor'] = svm_regressor
        
        # Evaluate models
        results = {}
        
        # Classification metrics
        for name, model in [('rf_classifier', rf_classifier), ('svm_classifier', svm_classifier)]:
            y_pred = model.predict(X_val)
            results[f'{name}_accuracy'] = accuracy_score(y_class_val, y_pred)
            results[f'{name}_f1'] = f1_score(y_class_val, y_pred)
        
        # Regression metrics
        for name, model in [('rf_regressor', rf_regressor), ('svm_regressor', svm_regressor)]:
            y_pred = model.predict(X_val)
            results[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_reg_val, y_pred))
        
        return results
    
    def train_tinyml_models(self, X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val):
        """Train TinyML-style models using scikit-learn"""
        print("Training TinyML-style models...")
        
        input_shape = X_train.shape[1]
        
        # MLP Classification
        mlp_classifier = self.create_mlp_model(input_shape, 'classification')
        mlp_classifier.fit(X_train, y_class_train)
        
        # MLP Regression
        mlp_regressor = self.create_mlp_model(input_shape, 'regression')
        mlp_regressor.fit(X_train, y_reg_train)
        
        # Store models
        self.models['mlp_classifier'] = mlp_classifier
        self.models['mlp_regressor'] = mlp_regressor
        
        # Evaluate models
        results = {}
        
        # Classification metrics
        y_pred = mlp_classifier.predict(X_val)
        results['mlp_classifier_accuracy'] = accuracy_score(y_class_val, y_pred)
        results['mlp_classifier_f1'] = f1_score(y_class_val, y_pred)
        
        # Regression metrics
        y_pred_reg = mlp_regressor.predict(X_val)
        results['mlp_regressor_rmse'] = np.sqrt(mean_squared_error(y_reg_val, y_pred_reg))
        
        return results
    
    def save_models(self, X_test):
        """Save all trained models"""
        os.makedirs('models', exist_ok=True)
        
        saved_models = {}
        
        # Save all models using joblib
        for model_name, model in self.models.items():
            model_path = f'models/{model_name}.pkl'
            joblib.dump(model, model_path)
            saved_models[model_name] = model_path
            print(f"Saved {model_name} to {model_path}")
        
        return saved_models
    
    def get_model_stats(self, saved_models):
        """Get model statistics for analysis"""
        stats = {}
        
        for model_name, model_path in saved_models.items():
            if os.path.exists(model_path):
                model_size = os.path.getsize(model_path) / 1024  # Size in KB
                stats[model_name] = {
                    'size_kb': model_size,
                    'tinyml_compatible': model_size < 100,  # Relaxed threshold for sklearn models
                    'model_type': 'TinyML-style' if 'mlp' in model_name else 'Traditional ML'
                }
        
        return stats

def train_all_models(processed_data):
    """Main function to train all models"""
    trainer = ModelTrainer()
    
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_class_train = processed_data['y_class_train']
    y_class_val = processed_data['y_class_val']
    y_reg_train = processed_data['y_reg_train']
    y_reg_val = processed_data['y_reg_val']
    
    # Train baseline models
    baseline_results = trainer.train_baseline_models(
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val
    )
    
    # Train TinyML models
    tinyml_results = trainer.train_tinyml_models(
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val
    )
    
    # Save models
    saved_models = trainer.save_models(X_test)
    
    # Get model statistics
    model_stats = trainer.get_model_stats(saved_models)
    
    # Combine results
    all_results = {**baseline_results, **tinyml_results}
    
    return trainer, all_results, model_stats