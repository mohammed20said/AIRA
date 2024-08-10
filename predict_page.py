import streamlit as st
import pickle
import numpy as np
import joblib
import pandas as pd

@st.cache_data
def load_Encoder():
    with open('encoder_columns.pkl', 'rb') as file:
        encoder = pickle.load(file)
    return encoder
@st.cache_data
def load_MLP_model():
    MLP_model = joblib.load('mlp_bn_model.joblib')
    return MLP_model

@st.cache_data
def load_XGBOOST_model():
    xgb_model = joblib.load('xgb_model.joblib')
    return xgb_model
@st.cache_data
def load_FNN_model():
    FNN_model = joblib.load('fnn_model.joblib')
    return FNN_model


encoder  = load_Encoder()
mlp_model  = load_MLP_model()
xgb_model = load_XGBOOST_model()
FNN_model = load_FNN_model()
@st.cache_data
def load_data():
    return pd.read_excel('data_Finel.xlsx')




def show_predict_page():
    st.title("risk Prediction")

    st.write("""### We need some information to predict the risk""")
    df = load_data()

    # Extract unique values from the specified columns
    unique_field_of_activity = tuple(df['Field of activity'].dropna().unique())
    unique_process_machine = tuple(df['Process / Machine / Equipement'].dropna().unique())
    
    
    

    field_of_activity=st.selectbox("field of activity : ",unique_field_of_activity)
    process_machine_equipement=st.selectbox("Process , Machine or Equipement : ",unique_process_machine)
    
    
    st.write("### Select the model to display predictions:")
    col1, col2 = st.columns([1, 2])  # Adjusted widths
    with col1:  # Moving radio buttons to the first column
        model_option = st.radio(
            "", 
            ['MLP', 'XGBOOST', 'FNN'], 
            index=1  # Default to XGBOOST
        )

    with col2:
        if model_option == 'MLP':
            with st.expander(f"{model_option} Model Explanation"):
                st.write("""
                    **MLP (Multi-Layer Perceptron)**: 
                    - A type of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer.
                    - Each node in a layer is connected to every node in the subsequent layer, making it a fully connected network.
                    - MLP is used for both regression and classification tasks, and it learns the mapping function from inputs to outputs using backpropagation with gradient descent.
                    - The model's flexibility allows it to capture complex patterns in the data, but it requires careful tuning of hyperparameters such as the number of layers, number of neurons, learning rate, and activation functions.
                """)

        elif model_option == 'XGBOOST':
            with st.expander(f"{model_option} Model Explanation"):
                st.write("""
                    **XGBoost (Extreme Gradient Boosting)**: 
                    - An efficient and scalable implementation of gradient boosting that uses decision trees as base learners.
                    - XGBoost combines the predictions of multiple weak learners (decision trees) to produce a powerful, ensemble model.
                    - It incorporates several enhancements over traditional gradient boosting, including regularization to prevent overfitting, handling of missing values, and parallel processing capabilities.
                    - XGBoost is widely used in both regression and classification tasks and is known for its accuracy, speed, and interpretability.
                    - Key hyperparameters to tune include learning rate, maximum depth of trees, and the number of trees.
                """)

        elif model_option == 'FNN':
            with st.expander(f"{model_option} Model Explanation"):
                st.write("""
                    **FNN (Feedforward Neural Network)**: 
                    - A type of neural network where the connections between the nodes do not form a cycle.
                    - FNNs consist of layers of neurons that process input data and pass it forward through the network until it reaches the output layer.
                    - The network learns the weights of the connections between neurons through a process called training, where the difference between the predicted and actual values is minimized.
                    - FNNs are commonly used in regression tasks and can approximate complex functions given sufficient data and network capacity.
                    - While simpler than recurrent neural networks, FNNs are powerful for many tasks where the data can be represented in a fixed-size input and output format.
                """)

    
    ok = st.button("Predict")
        
    predictions = {
        'MLP': None,
        'XGBOOST': None,
        'FNN': None
    }

    # When the "Predict" button is pressed
    if ok:
        # Filter the DataFrame based on the selected values
        filtered_df = df[
            (df['Field of activity'] == field_of_activity) & 
            (df['Process / Machine / Equipement'] == process_machine_equipement)
        ]
        
         # Check the length of the filtered DataFrame
        if len(filtered_df) > 0:
            
            # Transform the filtered DataFrame using the saved OneHotEncoder
            filtered_df_new_encoded = pd.get_dummies(filtered_df, columns=[
                'Field of activity', 'Process / Machine / Equipement', 'Risk Related', 'Risk Causes', 'Risk Effects'
            ])
            filtered_df_new_encoded = filtered_df_new_encoded.reindex(columns=encoder, fill_value=0)

        
            # Separate features and target
            X = filtered_df_new_encoded.drop(columns=['Criticality (Severity * Occurrence * Impact)']).astype(np.float32)   
            # Predict using all models
            predictions['MLP'] = mlp_model.predict(X)
            predictions['XGBOOST'] = xgb_model.predict(X)
            predictions['FNN'] = FNN_model.predict(X)
            # Add predictions to DataFrame
            filtered_df['Predicted Criticality with MLP'] = predictions['MLP']
            filtered_df['Predicted Criticality with XGBOOST'] = predictions['XGBOOST']
            filtered_df['Predicted Criticality with FNN'] = predictions['FNN']
            
            # Display radio buttons for model selection
            
            
            # Prepare the columns to display
            display_columns = [
                'Field of activity', 'Process / Machine / Equipement', 
                'Risk Related', 'Risk Causes', 'Risk Effects', f'Predicted Criticality with {model_option}'
            ]
            filtered_df_display = filtered_df[display_columns].sort_values(
                by=f'Predicted Criticality with {model_option}', ascending=False
            ).head(10)

            # Display the filtered DataFrame
            st.dataframe(filtered_df_display)
        else:
            # Display an error message
            st.error("No risks to display for the selected Field of activity and Process/Machine/Equipment.")
    else :
        st.error("Select parameters and click 'Predict' to display results.")