# app.py (UPDATED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/final_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure '../models/final_model.pkl' exists.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/01_cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.warning("Original dataset not found for visualization.")
        return None

# Main function
def main():
    # Title and description
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown("""
    This app predicts the risk of heart disease based on patient health metrics. 
    Enter the required information below and get an instant risk assessment.
    """)
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Risk Prediction", "Data Exploration", "About"])
    
    with tab1:
        if model:
            st.markdown('<h2 class="sub-header">Patient Health Information</h2>', unsafe_allow_html=True)
            
            # Create input form with three columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<h3 class="section-header">Personal Information</h3>', unsafe_allow_html=True)
                sex = st.radio("**Gender**", ["Male", "Female"], help="Select your biological sex")
                st.write("---")
                
                st.markdown('<h3 class="section-header">Heart Measurements</h3>', unsafe_allow_html=True)
                thalach = st.slider("**Maximum Heart Rate**", 60, 220, 150, 
                                  help="Your highest heart rate during exercise")
            
            with col2:
                st.markdown('<h3 class="section-header">Exercise Test Results</h3>', unsafe_allow_html=True)
                cp = st.selectbox("**Chest Pain Type**", [
                    "Typical Angina", 
                    "Atypical Angina", 
                    "Non-anginal Pain", 
                    "Asymptomatic"
                ], help="Type of chest pain experienced")
                
                exang = st.radio("**Exercise-Induced Chest Pain**", ["No", "Yes"], 
                               help="Do you get chest pain during exercise?")
                
                oldpeak = st.slider("**ST Depression during Exercise**", 0.0, 6.0, 1.0, 0.1,
                                  help="ECG ST segment depression induced by exercise")
            
            with col3:
                st.markdown('<h3 class="section-header">Medical Test Results</h3>', unsafe_allow_html=True)
                slope = st.selectbox("**ST Segment Slope**", [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ], help="Slope of the peak exercise ST segment")
                
                ca = st.slider("**Number of Major Vessels**", 0, 4, 0,
                             help="Number of major vessels colored by fluoroscopy (0-4)")
                
                thal = st.selectbox("**Thalassemia Result**", [
                    "Normal",
                    "Fixed Defect",
                    "Reversible Defect"
                ], help="Result of thalassemia test")
            
            # Convert categorical inputs to numerical values
            sex_num = 1 if sex == "Male" else 0
            cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
            exang_num = 1 if exang == "Yes" else 0
            slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
            
            # Create feature array in the EXACT ORDER your model expects
            input_features = np.array([[
                thalach,    # Max Heart Rate
                slope_num,  # Slope
                ca,         # Number of major vessels
                oldpeak,    # ST depression
                thal_num,   # Thalassemia
                cp_num,     # Chest Pain Type
                exang_num,  # Exercise Induced Angina
                sex_num     # Sex
            ]])
            
            # Large, prominent prediction button
            st.markdown("---")
            col_pred1, col_pred2, col_pred3 = st.columns([1,2,1])
            with col_pred2:
                if st.button("🔍 ANALYZE HEART DISEASE RISK", type="primary", 
                           use_container_width=True, help="Click to analyze your risk"):
                    try:
                        # Make prediction
                        prediction = model.predict(input_features)
                        prediction_proba = model.predict_proba(input_features)
                        
                        # Display results with large, clear formatting
                        st.markdown("---")
                        st.markdown('<h2 class="sub-header">📋 Prediction Results</h2>', unsafe_allow_html=True)
                        
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            if prediction[0] == 1:
                                st.markdown('<div class="prediction-result" style="background-color: #ffebee; color: #c62828;">🚨 HIGH RISK<br>Heart Disease Detected</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="prediction-result" style="background-color: #e8f5e8; color: #2e7d32;">✅ LOW RISK<br>No Heart Disease</div>', unsafe_allow_html=True)
                        
                        with result_col2:
                            risk_percentage = prediction_proba[0][1] * 100
                            st.metric("**Risk Probability**", f"{risk_percentage:.1f}%", 
                                    delta="High risk" if risk_percentage > 50 else "Low risk",
                                    delta_color="inverse")
                        
                        # Show probability distribution
                        fig, ax = plt.subplots(figsize=(10, 3))
                        bars = ax.barh(['Low Risk', 'High Risk'], 
                                    [prediction_proba[0][0], prediction_proba[0][1]], 
                                    color=['#4caf50', '#f44336'])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability', fontsize=12)
                        ax.set_title('Risk Probability Distribution', fontsize=14, fontweight='bold')
                        
                        # Add value labels on bars
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                  f'{width:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {e}")
    
    with tab2:
        st.header("Data Exploration")
        
        if df is not None:
            # Show basic dataset info
            st.subheader("Dataset Overview")
            st.write(f"Dataset shape: {df.shape}")
            st.dataframe(df.head())
            
            # Show only the selected features
            st.subheader("Selected Features Analysis")
            selected_cols = ['thalach', 'slope', 'ca', 'oldpeak', 'thal', 'cp', 'exang', 'sex', 'target']
            selected_df = df[selected_cols]
            st.write("Features used in the model:", selected_cols[:-1])
            st.dataframe(selected_df.describe())
            
            # Interactive visualizations
            st.subheader("Interactive Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Max Heart Rate distribution by target
                fig1 = px.histogram(df, x='thalach', color='target', 
                                   title='Max Heart Rate by Heart Disease Status',
                                   labels={'target': 'Heart Disease', 'thalach': 'Max Heart Rate'},
                                   color_discrete_map={0: 'green', 1: 'red'})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Chest Pain Type vs Heart Disease
                fig2 = px.histogram(df, x='cp', color='target',
                                   title='Chest Pain Type vs Heart Disease',
                                   labels={'cp': 'Chest Pain Type', 'target': 'Heart Disease'},
                                   color_discrete_map={0: 'green', 1: 'red'})
                st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation heatmap of selected features
            st.subheader("Correlation of Selected Features")
            numeric_selected_df = selected_df.select_dtypes(include=[np.number])
            fig3 = px.imshow(numeric_selected_df.corr(), 
                            title='Correlation Matrix of Selected Features',
                            color_continuous_scale='RdBu_r',
                            aspect="auto")
            st.plotly_chart(fig3, use_container_width=True)
            
        else:
            st.info("Upload the heart disease dataset to enable data exploration features.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='font-size: 1.2rem; line-height: 1.6;'>
        <h3 style='color: #2c3e50;'>❤️ Heart Disease Risk Predictor</h3>
        
        <p><strong>Purpose:</strong> This application uses machine learning to assess your risk of heart disease 
        based on key medical parameters identified through advanced feature selection.</p>
        
        <h4 style='color: #3498db;'>🔍 Key Features Analyzed:</h4>
        <ul>
            <li>Maximum Heart Rate Achieved</li>
            <li>ST Segment Slope during exercise</li>
            <li>Number of major blood vessels</li>
            <li>ST Depression measurement</li>
            <li>Thalassemia test results</li>
            <li>Chest Pain Type</li>
            <li>Exercise-Induced Angina</li>
            <li>Biological Sex</li>
        </ul>
        
        <h4 style='color: #3498db;'>⚙️ Technical Details:</h4>
        <ul>
            <li>Built with Streamlit and Python</li>
            <li>Uses Random Forest machine learning algorithm</li>
            <li>Trained on UCI Heart Disease dataset</li>
            <li>Features selected through statistical analysis</li>
        </ul>
        
        <div style='background-color: #fff3cd; padding: 1rem; color: red ; border-radius: 5px; border-left: 4px solid #ffc107;'>
        ⚠️ <strong>Important Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare providers for medical concerns.
        </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()