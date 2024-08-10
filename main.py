import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from user import login, sign_up,update_password
from streamlit_navigation_bar import st_navbar

headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
signUpSection = st.container()
logOutSection = st.container()

def change_password():
    st.subheader("Change Password")

    old_password = st.text_input("Old Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_new_password = st.text_input("Confirm New Password", type="password")

    if st.button("Submit"):
        if not old_password or not new_password or not confirm_new_password:
            st.error("All fields are required.")
        elif len(new_password) < 8:
            st.error("New password must be at least 8 characters long.")
        elif new_password != confirm_new_password:
            st.error("New password and confirm password do not match.")
        else:
            # Assuming you have a function `update_password` to handle the update
            success = update_password(st.session_state['username'], old_password, new_password)
            if success:
                st.success("Password changed successfully!")
                
            else:
                st.error("Failed to change password. Please check your old password.")

def show_main_page():
    loginSection.empty()
    
    
    st.sidebar.title("AIRA")
    st.sidebar.markdown("""\n\n\n""")
    
    st.sidebar.button("Log Out", key="logout", on_click=LoggedOut_Clicked,type="primary")
    
    
    
    # Selection box for navigation
    page = st.sidebar.selectbox("Chose the action", ("Predict", "Explore","Change Password"))
    
    
    if page == "Predict":
        show_predict_page()
    elif page == "Explore":
        show_explore_page()
    elif page == "Change Password":
        change_password()
        
def show_home_page():
    st.sidebar.title("AIRA")
    st.sidebar.markdown("""\n\n\n""")
    if st.sidebar.button("Login"):
        st.session_state['show_signup'] = False
    if st.sidebar.button("Sign Up"):
        st.session_state['show_signup'] = True

    st.markdown("""
    # Welcome to AIRA
    AIRA is a cutting-edge application designed to help you with risk analysis using advanced AI models.
    - **Explore** the various features of the application.
    - **Predict** outcomes based on your input data.
    - **Analyze** the results and gain insights.
    """)
    
    st.markdown("""
    ### Models Used in AIRA
    - **FNN (Feedforward Neural Network):** A simple neural network architecture that helps in predicting outcomes by passing input data through multiple layers of neurons.
    - **MLP (Multi-Layer Perceptron):** A class of feedforward artificial neural networks that uses multiple layers to enhance prediction accuracy.
    - **XGBoost:** An optimized gradient boosting algorithm that excels in handling structured data and delivering top-tier results.
    
    _These models are integrated into AIRA to provide you with the most accurate and reliable risk analysis possible._
    """)

def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    st.session_state['show_signup'] = False

def show_login_page():
    with loginSection:
        if not st.session_state.get('loggedIn', False):
            userName = st.text_input("User Name", key="login_userName", placeholder="Enter your user name")
            password = st.text_input("Password", key="login_password", placeholder="Enter password", type="password")
            st.button("Login", key="login", on_click=lambda: LoggedIn_Clicked(userName, password))
            st.button("Don't have an account?", key="signup",on_click=lambda: dont_have_account_clicked())
            
def dont_have_account_clicked():
    st.session_state['show_signup'] = True
    st.session_state['loggedIn'] = False

def LoggedIn_Clicked(userName: str, password: str):
    if login(userName, password):
        st.session_state['loggedIn'] = True
        st.session_state['show_signup'] = False
        st.session_state['username'] = userName
    else:
        st.session_state['loggedIn'] = False
        st.error("Invalid user name or password")

def Signup_Clicked(userName, password, confirm_password):
    if not userName or not password:
        st.error("Username and password are required!")
    elif len(password) < 8:
        st.error("Password must be at least 8 characters long!")
    elif password != confirm_password:
        st.error("Passwords do not match!")
    elif sign_up(userName, password):
        st.success("Account created successfully! Please log in.")
        st.session_state['showSignup'] = False
    else:
        st.error("Sign-up failed. Please try again.")
    

def show_sign_up_page():
    with loginSection:
        userName = st.text_input("User Name", key="signup_userName", placeholder="Enter your user name")
        password = st.text_input("Password", key="signup_password", placeholder="Enter password", type="password")
        confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm_password",placeholder="Confirme password")
        st.button("Sign Up", key="signup_submit", on_click=lambda: Signup_Clicked(userName, password, confirm_password))
        st.button("Back to Login", key="back_to_login", on_click=lambda: back_login_clicked())
        
def back_login_clicked():
    st.session_state['show_signup'] = False
    st.session_state['loggedIn'] = False

def show_logout_page():
    loginSection.empty()
    with logOutSection:
        st.button("Log Out", key="logout", on_click=LoggedOut_Clicked)

with headerSection:
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        st.session_state['show_signup'] = False
        st.session_state['username'] = ""
        show_home_page()
    else:
        if st.session_state['loggedIn']:
            show_main_page()
        else:
            if st.session_state.get('show_signup', False):
                show_sign_up_page()
            else:
                show_login_page()