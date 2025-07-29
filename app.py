import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

warnings.filterwarnings("ignore")

with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

# Assume 'model' and 'encoder' are available globally or loaded here
# In a real application, you would load the trained model and encoder
# from saved files (e.g., using joblib or pickle).
# Since the notebook environment doesn't save files easily between runs,
# we will re-initialize them based on the training steps.

# Re-initialize LabelEncoder and fit it on the 'Extracurricular Activities' column
encoder = LabelEncoder()
# Assuming 'data' DataFrame is available globally from previous steps
# If not, you would need to reload it or pass it
# Load the data if it's not already in the environment
try:
    data_for_encoder = pd.read_csv('Student_Performance.csv')
    encoder.fit(data_for_encoder['Extracurricular Activities'])
except NameError:
    st.error("Data for encoder not found. Please ensure the data loading step ran successfully.")
    st.stop()


# Re-initialize and train the model
# Assuming X_train and y_train are available globally from previous steps
# If not, you would need to reload or re-split the data
try:
    model = LinearRegression()
    model.fit(X_train, y_train)
except NameError:
     st.error("Training data (X_train, y_train) not found. Please ensure the data splitting and model training steps ran successfully.")
     st.stop()


st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# Function for the prediction page
def show_predict_page():
    st.title("Predict Student Performance/Exam Score")

    st.write("""
    ### Enter student details to predict their performance index.
    """)

    hours_studied = st.number_input('Hours Studied', min_value=0, max_value=12, value=0)
    previous_scores = st.number_input('Previous Scores', min_value=0, max_value=99, value=0)
    extracurricular_activities = st.selectbox('Extracurricular Activities', ('Yes', 'No'))
    sleep_hours = st.number_input('Sleep Hours', min_value=0, max_value=9, value=0)
    sample_papers = st.number_input('Sample Question Papers Practiced', min_value=0, max_value=9, value=0)

    predict_button = st.button("Predict Score")

    if predict_button:
        input_data = pd.DataFrame({
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_scores],
            'Extracurricular Activities': [extracurricular_activities],
            'Sleep Hours': [sleep_hours],
            'Sample Question Papers Practiced': [sample_papers]
        })

        input_data['Extracurricular Activities'] = encoder.transform(input_data['Extracurricular Activities'])

        prediction = model.predict(input_data)   

        # Ensure the predicted score fall within the range of 0 to 100
        predicted_score = prediction[0]
        if predicted_score > 100:
            predicted_score = 100
        elif predicted_score < 0:
            predicted_score = 0 

        # Compute confidence interval (95%)
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        se = np.sqrt(np.sum(residuals**2) / (len(y_train) - 2))
        mean_x = np.mean(X_train, axis=0)
        n = len(y_train)
        t_value = 2.0  # Approx for large df, 95% CI

        se_pred = se * np.sqrt(1 + (1/n) + ((input_data - mean_x)**2).sum(axis=1) / ((X_train - mean_x)**2).sum().sum())
        lower_bound = predicted_score - t_value * se_pred[0]
        if lower_bound < 0:
            lower_bound = 0
        upper_bound = predicted_score + t_value * se_pred[0]
        if upper_bound > 100:
            upper_bound = 100

        st.subheader(f"The predicted score is: {np.round(predicted_score, decimals=0)}")
        st.markdown(f"#### The score is expected to fall within the 95% confidence interval of **{lower_bound:.2f}** to **{upper_bound:.2f}**.")
        
# Function for the information page
def show_info_page():
    st.title("About This Web App")

    st.header("Problem Statement")
    st.write("""
    Predicting student exam scores is a crucial task for educational institutions (an example is HTU).
    Accurate predictions can help identify students at risk, personalize learning support, and improve overall
    academic outcomes. However, understanding the factors influencing performance and
    building a reliable predictive model can be challenging. This web application aims to address this challenge
    by providing a tool to predict student performance based on several key indicators.
    """)

    st.header("Objectives of the Web App")
    st.write("""
    *   To provide a user-friendly interface for predicting student exam scores.
    *   To demonstrate the factors that significantly influence student performance.
    *   To offer insights that can help students, educators, and administrators understand
        and potentially improve academic outcomes.
    """)

    st.header("Methodology Used")
    st.write("""
    This web application uses a **Multiple Linear Regression** model to predict student performance.
    Linear Regression is a Machine Learning and a statistical approach that models the relationship between a dependent
    variable (Performance Index) and one or more independent variables (features) by fitting a
    linear equation to the observed data. The model was trained on historical student data to
    learn the coefficients that best describe the linear relationship between the input features
    and the performance index.
    """)

    st.header("Description of Data Variables")
    st.write("""
    *   **Hours Studied:** The total number of hours a student spent studying for the exam (it may be just a course).
    *   **Previous Scores:** The average score of the student in previous exams/or the previous score of the student in a related course.
    *   **Extracurricular Activities:** Whether the student participates in extracurricular activities (Yes/No). Examples are coding clubs, regular educational activities, etc.
    *   **Sleep Hours:** The average number of hours a student sleeps per night before the exam.
    *   **Sample Question Papers Practiced:** The number of sample question papers the student practiced (it could be past questions).
    *   **Performance Index (Target Variable):** The predicted score of the student in the exam (on a scale, typically 0-100). This can also be the predicted average score of all courses being taken for a particular semester. Provided that the previous score entered by the student is an average of all the scores of the courses taken in the previous semester.  
    """)

    st.header("Key Findings from Data Analysis")
    st.write("""
    Based on the analysis of the dataset:
    *   **Hours Studied** and **Previous Scores** have the strongest positive correlation with the Performance Index.
    *   Participating in **Extracurricular Activities** shows a small positive impact on performance.
    *   **Sleep Hours** also has a little positive correlation with performance.
    *   The number of **Sample Question Papers Practiced** has a positive but relatively weaker correlation compared to Hours Studied and Previous Scores.
    *   There were no missing values in the dataset, and while there were duplicate rows, they likely represent students with identical characteristics rather than true data entry errors.
    """)

    st.header("Recommendations")
    st.write("""
    Based on the model and analysis:
    *   **Prioritize Study Time:** Students should focus on increasing their study hours as it is a significant predictor of performance.
    *   **Build on Previous Knowledge:** A strong foundation from previous studies (reflected in Previous Scores) is crucial.
    *   **Consider Extracurriculars:** While the impact is small, extracurricular activities might contribute positively to overall student development and performance.
    *   **Ensure Adequate Sleep:** Sufficient sleep is positively correlated with better performance.
    *   **Practice with Sample Papers:** Practicing sample papers, although less impactful than study hours and previous scores, can still be beneficial.
    """)

# --- Streamlit App Navigation ---
page = st.sidebar.selectbox("Navigation Tab", ["Prediction", "About"])

if page == "Prediction":
    show_predict_page()
elif page == "About":
    show_info_page()
