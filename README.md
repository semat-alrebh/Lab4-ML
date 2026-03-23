# Lab-4-ML

# ARTI308 – Lab 4: Data Quality Assessment & Preprocessing

---

## Project Overview
This project applies Data Quality Assessment and Preprocessing techniques to an AI Job Market dataset.  
The goal is to clean, transform, and prepare the data properly before applying any machine learning model.

---

## Dataset
The dataset used in this project is:

**AI Job Market Dataset (AIJobMarket.csv)**

The dataset contains information about AI-related job postings, including:
- Job title and job ID
- Company size and industry
- Country and remote type
- Experience level and years of experience
- Education level
- Skills (Python, SQL, ML, Deep Learning, Cloud)
- Salary
- Job posting month and year
- Hiring urgency and job openings

---

## Preprocessing Techniques Applied

The following steps were performed:

- **Task 1 – Data Quality Assessment:**
  - Checked data types using `dtypes` and `info()`
  - Converted categorical columns from `object` to `category` dtype
  - Checked for duplicate rows
  - Identified potential outliers in the `salary` column

- **Task 2 – Handling Missing Values:**
  - Detected missing values using `isna().sum()`
  - Introduced artificial missing values for learning purposes
  - Applied **Median Imputation** to fill missing salary values
  - Chose median over mean due to the skewed salary distribution

- **Task 3 – Handling Outliers:**
  - Visualized the `salary` column using a boxplot
  - Detected outliers using the **IQR method** (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
  - Applied **Capping (5th–95th Percentile)** to handle extreme values
  - Preserved all records since extreme salaries are valid real-world data

- **Task 4 – Normalization:**
  - Applied **Min-Max Normalization** to scale values between 0 and 1
  - Applied **Z-Score Standardization** to center data around mean=0, std=1
  - Both techniques applied to `salary` and `years_experience` columns

- **Task 5 – PCA (Principal Component Analysis):**
  - Checked correlation between `salary` and `years_experience`
  - Correlation was approximately **-0.013** (near zero)
  - Concluded that PCA is **not beneficial** for these features
  - Applied PCA for demonstration purposes only

---

## Files Included
- `AIJobMarket.csv` → Dataset file  
- `Lab4ML.ipynb` → Jupyter Notebook containing the full analysis  

---

## Conclusion
Through this lab, key preprocessing techniques were applied to the AI Job Market dataset.  
The analysis showed that salary data contains outliers that are best handled by capping rather than removal, since extreme salaries represent valid high-level positions.  
Normalization was applied to prepare numerical features for machine learning models, and PCA was explored but found to be unnecessary due to the low correlation between features.
