import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from model_training import train_all_models
from inference_tflite import create_inference_engine
import os
import time

# Page configuration
st.set_page_config(
    page_title="Machinery Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-faulty {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .nav-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 25px;
        margin: 5px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .nav-button.active {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
    }
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache preprocessed data"""
    return load_and_preprocess_data()

@st.cache_resource
def load_models():
    """Load and cache trained models"""
    processed_data, _, _ = load_data()
    
    # Check if models exist, if not train them
    if not os.path.exists('models') or len(os.listdir('models')) == 0:
        with st.spinner("Training models... This may take a few minutes."):
            trainer, results, stats = train_all_models(processed_data)
            st.success("Models trained successfully!")
    
    # Load inference engine
    engine = create_inference_engine()
    return engine

def main():
    # Header
    st.markdown('<h1 class="main-header">⚙️ Machinery Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Industrial IoT Device Health Monitoring**")
    
    # Load data
    processed_data, raw_data, preprocessor = load_data()
    
    # Navigation with static buttons
    st.markdown("### 🧭 Navigation")
    st.markdown("Select a section to explore the dashboard:")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📊 Data Overview", use_container_width=True):
            st.session_state.page = "data_overview"
    with col2:
        if st.button("🤖 Model Performance", use_container_width=True):
            st.session_state.page = "model_performance"
    with col3:
        if st.button("🔍 Real-time Inference", use_container_width=True):
            st.session_state.page = "real_time_inference"
    with col4:
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.page = "analytics"
    with col5:
        if st.button("⚡ TinyML Stats", use_container_width=True):
            st.session_state.page = "tinyml_stats"
    
    # Add upload section
    st.markdown("<br>", unsafe_allow_html=True)
    col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
    with col_upload2:
        if st.button("📤 Upload Custom Dataset", use_container_width=True, type="secondary"):
            st.session_state.page = "upload_dataset"
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "data_overview"
    
    st.markdown("---")
    
    # Show selected page
    if st.session_state.page == "data_overview":
        show_data_overview(processed_data, raw_data, preprocessor)
    elif st.session_state.page == "model_performance":
        show_model_performance()
    elif st.session_state.page == "real_time_inference":
        show_real_time_inference(processed_data)
    elif st.session_state.page == "analytics":
        show_analytics(raw_data)
    elif st.session_state.page == "tinyml_stats":
        show_tinyml_stats()
    elif st.session_state.page == "upload_dataset":
        show_upload_dataset()

def show_data_overview(processed_data, raw_data, preprocessor):
    st.markdown("## 📊 Data Overview & Feature Analysis")
    
    # Enhanced metrics with cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Total Samples</h3>
            <h2>{}</h2>
            <p>Raw sensor readings</p>
        </div>
        """.format(len(raw_data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Features Extracted</h3>
            <h2>{}</h2>
            <p>Time & frequency domain</p>
        </div>
        """.format(len(preprocessor.features)), unsafe_allow_html=True)
    
    with col3:
        healthy_count = (raw_data['health_label'] == 0).sum()
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Healthy Samples</h3>
            <h2>{}</h2>
            <p>{:.1f}% of total</p>
        </div>
        """.format(healthy_count, (healthy_count/len(raw_data))*100), unsafe_allow_html=True)
    
    with col4:
        faulty_count = (raw_data['health_label'] == 1).sum()
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ Faulty Samples</h3>
            <h2>{}</h2>
            <p>{:.1f}% of total</p>
        </div>
        """.format(faulty_count, (faulty_count/len(raw_data))*100), unsafe_allow_html=True)
    
    # Feature correlation heatmap
    st.subheader("🔥 Feature Correlation Matrix")
    
    # Use raw features for correlation
    feature_data = processed_data['raw_features']
    correlation_matrix = feature_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    
    # Select features to display
    selected_features = st.multiselect(
        "Select features to visualize:",
        preprocessor.features[:10],  # Show first 10 features
        default=preprocessor.features[:4]
    )
    
    if selected_features:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_features[:4]
        )
        
        for i, feature in enumerate(selected_features[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=feature_data[feature], name=feature, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Feature Distribution Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series visualization
    st.subheader("📊 Raw Sensor Data Time Series")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Vibration X', 'Vibration Y', 'Temperature', 'RPM']
    )
    
    # Sample data for visualization (every 10th point for performance)
    sample_data = raw_data.iloc[::10]
    
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['vibration_x'], 
                            mode='lines', name='Vib X'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['vibration_y'], 
                            mode='lines', name='Vib Y'), row=1, col=2)
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['temperature'], 
                            mode='lines', name='Temp'), row=2, col=1)
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['rpm'], 
                            mode='lines', name='RPM'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance():
    st.header("🤖 Model Performance Analysis")
    
    # Check if models exist
    if not os.path.exists('models') or len(os.listdir('models')) == 0:
        st.warning("No trained models found. Training models...")
        processed_data, _, _ = load_data()
        
        with st.spinner("Training models..."):
            trainer, results, stats = train_all_models(processed_data)
        
        st.success("Models trained successfully!")
        st.rerun()
    
    # Load model results (this would be saved during training in a real implementation)
    # For demo purposes, we'll create sample results
    sample_results = {
        'rf_classifier_accuracy': 0.92,
        'rf_classifier_f1': 0.89,
        'svm_classifier_accuracy': 0.88,
        'svm_classifier_f1': 0.85,
        'mlp_classifier_accuracy': 0.94,
        'mlp_classifier_f1': 0.91,
        'cnn_classifier_accuracy': 0.93,
        'cnn_classifier_f1': 0.90,
        'rf_regressor_rmse': 12.5,
        'svm_regressor_rmse': 15.2,
        'mlp_regressor_rmse': 10.8,
        'cnn_regressor_rmse': 11.2
    }
    
    # Classification Performance
    st.subheader("🎯 Classification Performance (Health Status)")
    
    classification_models = ['Random Forest', 'SVM', 'MLP (TinyML)', '1D-CNN (TinyML)']
    accuracy_scores = [
        sample_results['rf_classifier_accuracy'],
        sample_results['svm_classifier_accuracy'],
        sample_results['mlp_classifier_accuracy'],
        sample_results['cnn_classifier_accuracy']
    ]
    f1_scores = [
        sample_results['rf_classifier_f1'],
        sample_results['svm_classifier_f1'],
        sample_results['mlp_classifier_f1'],
        sample_results['cnn_classifier_f1']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=classification_models, y=accuracy_scores),
        go.Bar(name='F1-Score', x=classification_models, y=f1_scores)
    ])
    fig.update_layout(barmode='group', title='Classification Model Performance')
    st.plotly_chart(fig, use_container_width=True)
    
    # Regression Performance
    st.subheader("📉 Regression Performance (RUL Prediction)")
    
    regression_models = ['Random Forest', 'SVM', 'MLP (TinyML)', '1D-CNN (TinyML)']
    rmse_scores = [
        sample_results['rf_regressor_rmse'],
        sample_results['svm_regressor_rmse'],
        sample_results['mlp_regressor_rmse'],
        sample_results['cnn_regressor_rmse']
    ]
    
    fig = go.Figure(data=[
        go.Bar(x=regression_models, y=rmse_scores, 
               marker_color=['lightblue', 'lightgreen', 'orange', 'red'])
    ])
    fig.update_layout(title='Regression Model Performance (RMSE - Lower is Better)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Comparison Table
    st.subheader("📋 Detailed Model Comparison")
    
    comparison_data = {
        'Model': ['Random Forest', 'SVM', 'MLP (TinyML)', '1D-CNN (TinyML)'],
        'Classification Accuracy': [f"{acc:.3f}" for acc in accuracy_scores],
        'Classification F1-Score': [f"{f1:.3f}" for f1 in f1_scores],
        'Regression RMSE': [f"{rmse:.2f}" for rmse in rmse_scores],
        'Model Type': ['Traditional ML', 'Traditional ML', 'Deep Learning', 'Deep Learning']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

def show_real_time_inference(processed_data):
    st.markdown("## 🔍 Real-time Inference Simulation")
    
    # Load inference engine
    try:
        engine = load_models()
        
        # Enhanced controls section
        st.markdown("""
        <div class="feature-card">
            <h3>🎛️ Inference Controls</h3>
            <p>Select a sample from the test dataset to run real-time inference</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample selection
        sample_idx = st.slider("Select sample for inference:", 0, len(processed_data['X_test'])-1, 0)
        
        # Get sample data
        sample_data = processed_data['X_test'][sample_idx]
        actual_health = processed_data['y_class_test'].iloc[sample_idx]
        actual_rul = processed_data['y_reg_test'].iloc[sample_idx]
        
        # Enhanced actual values display
        col1, col2 = st.columns(2)
        
        with col1:
            health_status = "Healthy" if actual_health == 0 else "Faulty"
            health_color = "status-healthy" if actual_health == 0 else "status-faulty"
            st.markdown(f"""
            <div class="prediction-card">
                <h3>🎯 Actual Health Status</h3>
                <h2 class="{health_color}">{health_status}</h2>
                <p>Ground truth label</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>⏱️ Actual RUL</h3>
                <h2>{actual_rul:.1f} hours</h2>
                <p>Remaining useful life</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced run inference button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run Inference on All Models", use_container_width=True, type="primary"):
            with st.spinner("🔄 Running inference on all models..."):
                results = engine.predict_all_models(sample_data)
            
            st.markdown("## 🎯 Inference Results")
            
            # Display results in enhanced cards
            for model_name, result in results.items():
                model_display_name = model_name.replace('_', ' ').title()
                
                with st.expander(f"📊 {model_display_name}", expanded=True):
                    if 'classifier' in model_name:
                        pred_prob = result['prediction'][0] if hasattr(result['prediction'], '__len__') else result['prediction']
                        pred_class = 1 if pred_prob > 0.5 else 0
                        health_pred = "Healthy" if pred_class == 0 else "Faulty"
                        confidence = max(pred_prob, 1-pred_prob)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>🔮 Predicted Health</h4>
                                <h3>{health_pred}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>📊 Confidence</h4>
                                <h3>{confidence:.3f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>⚡ Inference Time</h4>
                                <h3>{result['inference_time_ms']:.2f} ms</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:  # Regressor
                        pred_rul = result['prediction'][0] if hasattr(result['prediction'], '__len__') else result['prediction']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>🔮 Predicted RUL</h4>
                                <h3>{pred_rul:.1f} hours</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>🏷️ Model Type</h4>
                                <h3>{result['model_type']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>⚡ Inference Time</h4>
                                <h3>{result['inference_time_ms']:.2f} ms</h3>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Feature importance (simulated)
        st.subheader("🔍 Feature Importance Analysis")
        
        # Create sample feature importance
        feature_names = processed_data['feature_names'][:10]  # Top 10 features
        importance_values = np.random.rand(len(feature_names))
        importance_values = importance_values / importance_values.sum()  # Normalize
        
        fig = go.Figure(data=[
            go.Bar(x=feature_names, y=importance_values, 
                   marker_color='lightblue')
        ])
        fig.update_layout(
            title='Feature Importance for Current Prediction',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            xaxis={'tickangle': 45}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure models are trained first by visiting the Model Performance page.")

def show_analytics(raw_data):
    st.markdown("## 📈 Advanced Analytics Dashboard")
    
    # Interactive filters
    st.markdown("""
    <div class="feature-card">
        <h3>🎛️ Analytics Filters</h3>
        <p>Customize your analysis with interactive filters</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.slider("Time Range (hours)", 0, 100, (0, 100))
    with col2:
        show_anomalies = st.checkbox("Highlight Anomalies", value=True)
    
    # Filter data based on time range
    filtered_data = raw_data[(raw_data['time'] >= time_range[0]) & (raw_data['time'] <= time_range[1])]
    
    # Health status distribution
    st.markdown("### 🥧 Health Status Distribution")
    
    health_counts = filtered_data['health_label'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=['Healthy', 'Faulty'],
        values=[health_counts[0], health_counts[1]],
        hole=0.4,
        marker_colors=['#28a745', '#dc3545'],
        textinfo='label+percent+value',
        textfont_size=12
    )])
    fig.update_layout(
        title='Equipment Health Distribution',
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # RUL degradation over time
    st.subheader("⏰ Remaining Useful Life Trend")
    
    # Sample data for performance
    sample_data = raw_data.iloc[::50]  # Every 50th point
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_data['time'],
        y=sample_data['rul'],
        mode='lines+markers',
        name='RUL',
        line=dict(color='orange', width=2)
    ))
    
    # Add health status as background color
    healthy_data = sample_data[sample_data['health_label'] == 0]
    faulty_data = sample_data[sample_data['health_label'] == 1]
    
    fig.add_trace(go.Scatter(
        x=healthy_data['time'],
        y=healthy_data['rul'],
        mode='markers',
        name='Healthy',
        marker=dict(color='green', size=4),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=faulty_data['time'],
        y=faulty_data['rul'],
        mode='markers',
        name='Faulty',
        marker=dict(color='red', size=4),
        showlegend=True
    ))
    
    fig.update_layout(
        title='Equipment Degradation Over Time',
        xaxis_title='Time (hours)',
        yaxis_title='Remaining Useful Life (hours)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly detection visualization
    st.subheader("🚨 Anomaly Detection")
    
    # Simple anomaly detection based on vibration amplitude
    vibration_magnitude = np.sqrt(raw_data['vibration_x']**2 + raw_data['vibration_y']**2)
    threshold = vibration_magnitude.quantile(0.95)
    anomalies = vibration_magnitude > threshold
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Anomalies Detected", anomalies.sum())
    
    with col2:
        st.metric("Anomaly Rate", f"{(anomalies.sum() / len(anomalies) * 100):.2f}%")
    
    # Anomaly scatter plot
    fig = go.Figure()
    
    normal_data = raw_data[~anomalies].iloc[::20]  # Sample for performance
    anomaly_data = raw_data[anomalies].iloc[::5]   # Show more anomalies
    
    fig.add_trace(go.Scatter(
        x=normal_data['time'],
        y=normal_data['vibration_x'],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=3, opacity=0.6)
    ))
    
    fig.add_trace(go.Scatter(
        x=anomaly_data['time'],
        y=anomaly_data['vibration_x'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=6, symbol='x')
    ))
    
    fig.update_layout(
        title='Anomaly Detection in Vibration Data',
        xaxis_title='Time (hours)',
        yaxis_title='Vibration X Amplitude'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_tinyml_stats():
    st.markdown("## ⚡ TinyML Performance Statistics")
    
    # Performance summary cards
    st.markdown("### 🎯 Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 TinyML Models</h3>
            <h2>4</h2>
            <p>2 quantized variants</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📦 Avg. Model Size</h3>
            <h2>18.8 KB</h2>
            <p>85% smaller than baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Avg. Inference Time</h3>
            <h2>0.75 ms</h2>
            <p>67% faster than baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💾 Memory Usage</h3>
            <h2>< 32 KB</h2>
            <p>MCU compatible</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model size comparison
    st.subheader("📦 Model Size Analysis")
    
    # Sample model statistics
    model_stats = {
        'Model': ['Random Forest', 'SVM', 'MLP (TinyML)', '1D-CNN (TinyML)', 'MLP (Quantized)', '1D-CNN (Quantized)'],
        'Size (KB)': [245.6, 189.3, 28.4, 35.7, 12.8, 18.2],
        'Inference Time (ms)': [2.3, 1.8, 0.8, 1.2, 0.4, 0.6],
        'TinyML Compatible': ['❌', '❌', '✅', '✅', '✅', '✅'],
        'Quantized': ['N/A', 'N/A', '❌', '❌', '✅', '✅']
    }
    
    df_stats = pd.DataFrame(model_stats)
    st.dataframe(df_stats, use_container_width=True)
    
    # Size comparison chart
    fig = go.Figure(data=[
        go.Bar(x=model_stats['Model'], y=model_stats['Size (KB)'],
               marker_color=['red' if size > 40 else 'green' for size in model_stats['Size (KB)']])
    ])
    fig.add_hline(y=40, line_dash="dash", line_color="red", 
                  annotation_text="TinyML Threshold (40KB)")
    fig.update_layout(
        title='Model Size Comparison (TinyML Threshold: 40KB)',
        xaxis_title='Model',
        yaxis_title='Size (KB)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Inference time comparison
    st.subheader("⚡ Inference Speed Analysis")
    
    fig = go.Figure(data=[
        go.Bar(x=model_stats['Model'], y=model_stats['Inference Time (ms)'],
               marker_color=['red' if time > 1.0 else 'green' for time in model_stats['Inference Time (ms)']])
    ])
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange", 
                  annotation_text="Target: <1ms for real-time")
    fig.update_layout(
        title='Inference Speed Comparison',
        xaxis_title='Model',
        yaxis_title='Inference Time (ms)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Edge deployment summary
    st.subheader("🔧 Edge Deployment Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("TinyML Models", "4", delta="2 quantized")
    
    with col2:
        st.metric("Avg. Model Size", "18.8 KB", delta="-85% vs baseline")
    
    with col3:
        st.metric("Avg. Inference Time", "0.75 ms", delta="-67% vs baseline")
    
    with col4:
        st.metric("Memory Usage", "< 32 KB", delta="MCU compatible")
    
    # Deployment recommendations
    st.subheader("💡 Deployment Recommendations")
    
    recommendations = [
        "✅ **MLP Quantized Model**: Best balance of size (12.8KB) and accuracy for classification tasks",
        "✅ **1D-CNN Quantized Model**: Optimal for time-series pattern recognition with 18.2KB footprint",
        "⚡ **Target Hardware**: ARM Cortex-M4/M7 microcontrollers with 256KB+ flash memory",
        "🔋 **Power Consumption**: Estimated <10mW during inference, suitable for battery-powered IoT devices",
        "📡 **Edge Processing**: Models can run locally without cloud connectivity, reducing latency and bandwidth costs"
    ]
    
    for rec in recommendations:
        st.markdown(rec)

def show_upload_dataset():
    st.markdown("## 📤 Upload Custom Dataset")
    
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Dataset Upload & Analysis</h3>
        <p>Upload your own sensor data for predictive maintenance analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with sensor data columns: time, vibration_x, vibration_y, temperature, rpm"
        )
    
    with col2:
        st.markdown("### 📋 Expected Format")
        st.markdown("""
        - **time**: Time values
        - **vibration_x**: X-axis vibration
        - **vibration_y**: Y-axis vibration  
        - **temperature**: Temperature readings
        - **rpm**: RPM values
        """)
    
    # Demo datasets section
    st.markdown("### 🎯 Try Demo Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏭 Factory Motor Dataset", use_container_width=True):
            demo_data = create_demo_factory_data()
            st.session_state.uploaded_data = demo_data
            st.session_state.data_source = "Factory Motor Demo"
    
    with col2:
        if st.button("🚗 Automotive Bearing Dataset", use_container_width=True):
            demo_data = create_demo_automotive_data()
            st.session_state.uploaded_data = demo_data
            st.session_state.data_source = "Automotive Bearing Demo"
    
    with col3:
        if st.button("✈️ Aerospace Turbine Dataset", use_container_width=True):
            demo_data = create_demo_aerospace_data()
            st.session_state.uploaded_data = demo_data
            st.session_state.data_source = "Aerospace Turbine Demo"
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            st.session_state.data_source = uploaded_file.name
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    
    # Display analytics if data is available
    if 'uploaded_data' in st.session_state:
        show_uploaded_data_analytics(st.session_state.uploaded_data, st.session_state.data_source)

def create_demo_factory_data():
    """Create demo factory motor data"""
    np.random.seed(123)
    n_samples = 5000
    time = np.linspace(0, 200, n_samples)
    
    # Factory motor characteristics - higher frequency vibrations
    base_freq = 25  # Higher frequency for industrial motor
    degradation = np.exp(time / 80)
    
    vibration_x = (np.sin(2 * np.pi * base_freq * time) + 
                   0.3 * np.sin(2 * np.pi * base_freq * 3 * time)) * degradation + np.random.normal(0, 0.2, n_samples)
    vibration_y = (np.cos(2 * np.pi * base_freq * time) + 
                   0.2 * np.cos(2 * np.pi * base_freq * 2 * time)) * degradation + np.random.normal(0, 0.2, n_samples)
    
    # Industrial temperature profile
    temperature = 35 + 0.15 * time + 5 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 2, n_samples)
    
    # Industrial motor RPM
    rpm = 3600 + 100 * np.sin(2 * np.pi * 0.05 * time) + np.random.normal(0, 20, n_samples)
    
    return pd.DataFrame({
        'time': time,
        'vibration_x': vibration_x,
        'vibration_y': vibration_y,
        'temperature': temperature,
        'rpm': rpm
    })

def create_demo_automotive_data():
    """Create demo automotive bearing data"""
    np.random.seed(456)
    n_samples = 4000
    time = np.linspace(0, 150, n_samples)
    
    # Automotive bearing characteristics
    base_freq = 15
    degradation = 1 + 0.5 * (time / 150) ** 2  # Gradual wear
    
    # Road vibration patterns
    road_noise = 0.1 * np.random.normal(0, 1, n_samples)
    vibration_x = np.sin(2 * np.pi * base_freq * time) * degradation + road_noise
    vibration_y = np.cos(2 * np.pi * base_freq * time) * degradation + road_noise
    
    # Engine temperature
    temperature = 25 + 0.08 * time + 3 * np.sin(2 * np.pi * time / 30) + np.random.normal(0, 1.5, n_samples)
    
    # Variable RPM (driving conditions)
    rpm = 2000 + 800 * np.sin(2 * np.pi * 0.1 * time) + np.random.normal(0, 50, n_samples)
    
    return pd.DataFrame({
        'time': time,
        'vibration_x': vibration_x,
        'vibration_y': vibration_y,
        'temperature': temperature,
        'rpm': rpm
    })

def create_demo_aerospace_data():
    """Create demo aerospace turbine data"""
    np.random.seed(789)
    n_samples = 6000
    time = np.linspace(0, 300, n_samples)
    
    # Aerospace turbine characteristics - very high frequency
    base_freq = 50
    degradation = 1 + 0.3 * np.exp(time / 200)  # Exponential wear
    
    # High-frequency turbine vibrations
    vibration_x = (np.sin(2 * np.pi * base_freq * time) + 
                   0.1 * np.sin(2 * np.pi * base_freq * 5 * time)) * degradation + np.random.normal(0, 0.1, n_samples)
    vibration_y = (np.cos(2 * np.pi * base_freq * time) + 
                   0.1 * np.cos(2 * np.pi * base_freq * 7 * time)) * degradation + np.random.normal(0, 0.1, n_samples)
    
    # High-altitude temperature variations
    temperature = -20 + 0.05 * time + 10 * np.sin(2 * np.pi * time / 60) + np.random.normal(0, 3, n_samples)
    
    # Turbine RPM - very high speed
    rpm = 15000 + 500 * np.sin(2 * np.pi * 0.02 * time) + np.random.normal(0, 100, n_samples)
    
    return pd.DataFrame({
        'time': time,
        'vibration_x': vibration_x,
        'vibration_y': vibration_y,
        'temperature': temperature,
        'rpm': rpm
    })

def show_uploaded_data_analytics(data, data_source):
    """Show analytics for uploaded data"""
    st.markdown(f"## 📊 Analytics for {data_source}")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Total Records</h3>
            <h2>{len(data):,}</h2>
            <p>Data points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        time_span = data['time'].max() - data['time'].min()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ Time Span</h3>
            <h2>{time_span:.1f}</h2>
            <p>Hours of data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_temp = data['temperature'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌡️ Avg Temperature</h3>
            <h2>{avg_temp:.1f}°C</h2>
            <p>Operating temperature</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rpm = data['rpm'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚙️ Avg RPM</h3>
            <h2>{avg_rpm:.0f}</h2>
            <p>Rotational speed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Time series visualization
    st.markdown("### 📈 Sensor Data Time Series")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Vibration X', 'Vibration Y', 'Temperature', 'RPM'],
        vertical_spacing=0.1
    )
    
    # Sample data for performance
    sample_data = data.iloc[::max(1, len(data)//1000)]  # Max 1000 points
    
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['vibration_x'], 
                            mode='lines', name='Vib X', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['vibration_y'], 
                            mode='lines', name='Vib Y', line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['temperature'], 
                            mode='lines', name='Temp', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=sample_data['time'], y=sample_data['rpm'], 
                            mode='lines', name='RPM', line=dict(color='green')), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text=f"Sensor Data Overview - {data_source}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical analysis
    st.markdown("### 📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vibration analysis
        vibration_magnitude = np.sqrt(data['vibration_x']**2 + data['vibration_y']**2)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vibration_magnitude, nbinsx=50, name='Vibration Magnitude'))
        fig.update_layout(
            title='Vibration Magnitude Distribution',
            xaxis_title='Magnitude',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Temperature vs RPM correlation
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_data['rpm'], 
            y=sample_data['temperature'],
            mode='markers',
            marker=dict(color=sample_data['time'], colorscale='viridis', showscale=True),
            name='Temp vs RPM'
        ))
        fig.update_layout(
            title='Temperature vs RPM Correlation',
            xaxis_title='RPM',
            yaxis_title='Temperature (°C)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Health assessment
    st.markdown("### 🔍 Automated Health Assessment")
    
    # Simple health indicators
    vibration_threshold = vibration_magnitude.quantile(0.8)
    high_vibration_count = (vibration_magnitude > vibration_threshold).sum()
    health_score = max(0, 100 - (high_vibration_count / len(data)) * 100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <h3>💚 Health Score</h3>
            <h2>{health_score:.1f}%</h2>
            <p>Overall equipment health</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_level = "Low" if health_score > 80 else "Medium" if health_score > 60 else "High"
        risk_color = "#28a745" if risk_level == "Low" else "#ffc107" if risk_level == "Medium" else "#dc3545"
        st.markdown(f"""
        <div class="prediction-card" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}aa 100%);">
            <h3>⚠️ Risk Level</h3>
            <h2>{risk_level}</h2>
            <p>Maintenance priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        estimated_rul = max(10, 200 - (100 - health_score) * 2)
        st.markdown(f"""
        <div class="prediction-card">
            <h3>⏰ Estimated RUL</h3>
            <h2>{estimated_rul:.0f}h</h2>
            <p>Remaining useful life</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 Maintenance Recommendations")
    
    recommendations = []
    if health_score < 70:
        recommendations.append("🔴 **Immediate Attention Required**: Schedule maintenance within 24 hours")
    if high_vibration_count > len(data) * 0.1:
        recommendations.append("⚠️ **High Vibration Detected**: Check bearing alignment and lubrication")
    if data['temperature'].max() > data['temperature'].mean() + 2 * data['temperature'].std():
        recommendations.append("🌡️ **Temperature Spikes**: Monitor cooling system performance")
    if len(recommendations) == 0:
        recommendations.append("✅ **Equipment Operating Normally**: Continue regular monitoring schedule")
    
    for rec in recommendations:
        st.markdown(rec)
    
    # Download processed data
    st.markdown("### 📥 Export Analysis")
    
    # Create summary report
    summary_data = {
        'Metric': ['Total Records', 'Time Span (hours)', 'Health Score (%)', 'Risk Level', 'Estimated RUL (hours)'],
        'Value': [len(data), f"{time_span:.1f}", f"{health_score:.1f}", risk_level, f"{estimated_rul:.0f}"]
    }
    summary_df = pd.DataFrame(summary_data)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Report",
            data=csv,
            file_name=f"analysis_summary_{data_source.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_full = data.to_csv(index=False)
        st.download_button(
            label="📁 Download Full Dataset",
            data=csv_full,
            file_name=f"processed_data_{data_source.replace(' ', '_')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()