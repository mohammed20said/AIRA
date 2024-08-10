import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from predict_page import load_data,load_Encoder,load_FNN_model,load_MLP_model,load_XGBOOST_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df = load_data()
def general_statistics():
    st.write(
    """### Top 15 Fields of Activity by Mean Criticality
    """
    )
    # Group by 'Field of activity' and calculate the mean criticality
    mean_criticality = df.groupby('Field of activity')['Criticality (Severity * Occurrence * Impact)'].mean()

    # Sort the values and get the top 15
    top_15_activities = mean_criticality.sort_values(ascending=False).head(15)
    # Create a mapping of numbers to the 'Field of activity'
    activity_mapping = {f'Activity {i+1}': field for i, field in enumerate(top_15_activities.index)}
    # Prepare DataFrame for display
    activity_df = pd.DataFrame({
        'Number': list(activity_mapping.keys()),
        'Field of Activity': list(activity_mapping.values()),
        'Mean Criticality': top_15_activities.values
    })
    
    # Create columns for the layout
    col1, col2 = st.columns(2)

    # Left column for the plot
    with col1:
        
        
        # Plot the data with numbers instead of names
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.bar(activity_df['Number'], activity_df['Mean Criticality'], color='skyblue')
        ax.set_title('Top 15 Fields of Activity by Mean Criticality')
        ax.set_xlabel('Field of Activity')
        ax.set_ylabel('Mean Criticality')
        ax.set_xticklabels(activity_df['Number'], rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Right column for the DataFrame
    with col2:
        # Display the DataFrame in Streamlit
        st.dataframe(activity_df.style)
        
        
    
    
    
    st.write("### Top 15 Processes/Machines/Equipments by Mean Criticality.")
    # Group by 'Process / Machine / Equipement' and calculate the mean criticality
    mean_criticality1 = df.groupby('Process / Machine / Equipement')['Criticality (Severity * Occurrence * Impact)'].mean()

    # Sort the values and get the top 15
    top_15_processes = mean_criticality1.sort_values(ascending=False).head(15)

    # Create a mapping of numbers to the 'Process / Machine / Equipement'
    process_mapping = {f'Process {i+1}': process for i, process in enumerate(top_15_processes.index)}

    # Prepare DataFrame for display
    process_df = pd.DataFrame({
        'Number': list(process_mapping.keys()),
        'Process/Machine/Equipment': list(process_mapping.values()),
        'Mean Criticality': top_15_processes.values
    })

    # Create columns for the layout
    col1, col2 = st.columns(2)

    # Left column for the plot
    with col1:
        
        
        # Plot the data with numbers instead of names
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.bar(process_df['Number'], process_df['Mean Criticality'], color='skyblue')
        ax.set_title('Top 15 Processes/Machines/Equipments by Mean Criticality')
        ax.set_xlabel('Process/Machine/Equipment')
        ax.set_ylabel('Mean Criticality')
        ax.set_xticklabels(process_df['Number'], rotation=45, ha='right')
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

    # Right column for the DataFrame
    with col2:
       
        # Display the DataFrame in Streamlit
        st.dataframe(process_df)
    
    
    
def models_Result():
    encoder  = load_Encoder()
    # Encode categorical variables using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=[
        'Field of activity', 'Process / Machine / Equipement', 'Risk Related', 'Risk Causes', 'Risk Effects'
    ])
    df_encoded = df_encoded.reindex(columns=encoder, fill_value=0)

    # Separate features and target
    X = df_encoded.drop(columns=['Criticality (Severity * Occurrence * Impact)'])
    y = df_encoded['Criticality (Severity * Occurrence * Impact)']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    
    mlp_model  = load_MLP_model()
    xgb_model = load_XGBOOST_model()
    FNN_model = load_FNN_model()
    
    y_mlp = predict(mlp_model,X_test)
    y_xgboost = predict(xgb_model,X_test)
    y_FNN = predict(FNN_model,X_test)
    st.write(
            """### XGBoost Model Prediction Results """
            )
    col1, col2 = st.columns(2)
    with col1:
        # Plot the results
        plot_xgboost_results(y_test, y_xgboost)
        
    with col2 : 
        mae, mse, rmse, r2 = calculate_metrics(y_test, y_xgboost)
        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R-squared"],
            "Value": [mae, mse, rmse, r2]
        })
        
        st.table(metrics_df)
    st.write(
            """### MLP Model Prediction Results """
            )
    col1, col2 = st.columns(2)
    with col1:
        # Plot the results
        plot_xgboost_results(y_test, y_mlp)
        
    with col2 : 
        mae, mse, rmse, r2 = calculate_metrics(y_test, y_mlp)
        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R-squared"],
            "Value": [mae, mse, rmse, r2]
        })
        
        st.table(metrics_df)
    st.write(
            """### FNN Model Prediction Results """
            )
    col1, col2 = st.columns(2)
    with col1 :
        
        # Plot the results
        plot_xgboost_results(y_test, y_FNN)
    with col2 :
        mae, mse, rmse, r2 = calculate_metrics(y_test, y_FNN)
        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R-squared"],
            "Value": [mae, mse, rmse, r2]
        })
        
        st.table(metrics_df)
        

    
    
    
def show_explore_page():
    choice = st.sidebar.selectbox("Chose what to display : ",("General Statistics","Models Predictions vs Actuals"))
    if choice == "General Statistics":
        general_statistics()
    else :
        models_Result()
     
def predict(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred
  
def plot_xgboost_results(y_test, y_pred_xgb):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Scatter plot: Actual vs Predicted for XGBoost Model
    ax.scatter(y_test, y_pred_xgb.flatten(), color='blue', label='Predicted')
    ax.scatter(y_test, y_test, color='red', label='Actual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    
    # Labels and title
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('XGBoost Model')
    ax.legend()

    # Tight layout for better spacing
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2
    