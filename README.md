#Student Placement & Salary Prediction: A Machine Learning Study



## Project Overview

     This project explores a dataset of 100,000 student records to determine the predictability of placement outcomes and salary packages. Using a mix of  academic,extracurricular, and demographic features, I implemented a full Machine Learning pipeline from raw data cleaning to comparative model analysis.


## Key Objective: To establish a performance baseline for placement classification and identify whether academic metrics provide a strong enough signal for salary regression.


## Technical Workflow

    ### Data Cleaning & Preprocessing
           #### Dataset Scale: 100,000 rows with 26 feature columns.
           #### Cleaning: Handled missing values and performed data type conversion (stripping '%' symbols from string columns to enable float conversion).
           #### Feature Engineering:
                   Ordinal Mapping: Transformed college_tier (Tier 1, 2, 3) into numerical ranks.
                   One-Hot Encoding: Applied to categorical variables like branch and gender to prevent bias.
                   Scaling: Utilized StandardScaler to normalize feature distributions, ensuring distance-based algorithms performed optimally.
                
    ### Exploratory Data Analysis (EDA)
           #### The Correlation Mystery: Generated a correlation heatmap that revealed a "High-Entropy" environment. Most features showed a correlation of <0.05 with the target variables.  
           #### Feature Importance: Using Random Forest's built-in importance metrics, identified mock_interview_score and coding_skill_score as the primary (though  weak) drivers of the models.

    
## Model Performance & Comparison

    ### Classification (Placement Status)
    
            **Logistic Regression**
              Accuracy: **56.86%**
              Observations: **Baseline Winner.** Suggests a simple linear boundary is more effective than complex trees.
              
            **Random Forest (Default)**
                Accuracy: **55.64%**
                Observations: Suffered from high variance/overfitting due to deep tree growth.
                
            **Random Forest (Tuned)**
                Accuracy: **56.49%**
                Observations: Improved via pruning (max_depth=10), but hit the data's "noise ceiling."
               
    ### Regression (Salary Prediction)
            #### R² Score: 0.047
            #### Mean Absolute Error (MAE): 6.40 LPA
            #### Analysis: With an average salary of 7.25 LPA, an MAE of 6.40 confirms that salary distribution in this dataset is highly stochastic (random).

            
## Technical Conclusion: The "Noise Floor" Discovery 

    The most significant finding of this study is the identification of the Bayes Error Rate within the dataset.
    
    ### Signal vs. Noise: The consistency between Logistic Regression and Random Forest scores suggests that we have extracted the maximum possible signal from the 26 provided features.
    
    ### Predictive Limits: An accuracy of ~57% indicates that placement in this context is governed by variables not present in the dataset (e.g., soft skills, real-time interview performance, or specific company requirements).
    
    ### Model Integrity: I chose not to deploy a predictive interface (Streamlit) for active use, as the current model does not meet the reliability threshold (70%+) required for a professional student-facing tool.

    
