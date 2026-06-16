# **Mental Health Prediction Using Various ML/DL Methods**

---

## **Introduction**

With advances in Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), predictive systems can assist in identifying patterns and factors associated with individuals' likelihood of seeking mental health support. Such systems can provide valuable insights for healthcare professionals, employers, researchers, and policymakers in creating more supportive workplace environments and improving access to mental health resources.

This project focuses on analyzing and predicting employees' likelihood of seeking mental health treatment using survey data obtained from publicly available datasets. The study involves data cleaning, preprocessing, exploratory data analysis, and feature selection techniques to identify demographic, workplace, and psychological factors associated with treatment-seeking behavior.

Various Machine Learning and Deep Learning algorithms will be implemented and compared to determine the most effective model for predicting whether an employee is likely to seek mental health support. Model performance will be evaluated using different performance metrics to identify the most accurate and reliable predictive approach.

The findings from this study can provide insights into workplace mental health trends and support the development of targeted interventions, awareness programs, and data-driven policies aimed at improving employee mental health outcomes.

---

## **Problem Statement**

Mental health disorders have become a major global health concern, affecting individuals across different age groups, occupations, and social backgrounds. Mental health challenges continue to affect employee well-being, productivity, and quality of life across industries worldwide. Despite increased awareness, many individuals experiencing mental health difficulties do not seek professional suppor`t due to factors such as stigma, lack of workplace support, limited awareness, and accessibility barriers. Understanding the factors that influence mental health support-seeking behavior is essential for designing effective interventions and promoting employee well-being.

---
## **Aim and Objectives**

### Aim of the Study

The aim of this project is to develop and evaluate various Machine Learning and Deep Learning models for predicting the likelihood of employees seeking mental health support while identifying the most effective AI technique for accurate prediction.


### Objectives of the Study

1. To collect and preprocess employee mental health survey datasets for analysis.
2. To perform exploratory data analysis (EDA) to identify trends, patterns, and relationships among demographic, workplace, and behavioral variables.
3. To apply feature selection techniques to determine the most influential factors associated with mental health support-seeking behavior.
4. To implement various Machine Learning algorithms for predicting the likelihood of seeking mental health treatment.
5. To develop Deep Learning models and compare their performance with traditional Machine Learning approaches.
6. To evaluate the developed models using performance metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
7. To determine the most suitable predictive model based on model performance and reliability.
8. To provide insights that can assist employers, healthcare professionals, and policymakers in improving workplace mental health support systems and intervention strategies.

---
## **Materials & Methods**

### Research Design

This study utilizes a predictive quantitative research design grounded in data science, machine learning and deep learning. Supervised learning techniques are used in which machine learning algorithms are provided with training datasets that include features (input variables) alongside mental health treatment labels (output variables). This allows the models to create associations and provide accurate forecasts.

### Materials

#### Tools

Integrated Development Environment: Google Colaboratory
Programming Language: Python
Libraries:
- Pandas
- Matplotlib
- Seaborn
- Mumpy
- Sci-kit learn
- Tensorflow 

### Dataset

The data used for this analysis was purely secondary collected through Kaggle. Attribites of the dataset can be found below;

- Source: OSMI Mental Health in Tech Survey  
- Link: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey  
- Size: ~1,259 responses  
- Type: Structured survey data

### Models

#### Machine Learning Models
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting   

#### Deep Learning Model
- Artificial Neural Network (ANN)


### Workflow/Methodology

The project follows an end-to-end machine learning pipeline:

1. Data Cleaning and Preprocessing  
   - Handling missing values by eliminating rows with null values.
   - Standardizing categorical variables
   - Encoding categorical features  

2. Exploratory Data Analysis (EDA)  
   - Distribution analysis  
   - Feature relationships  
   - Class balance analysis  

3. Feature Engineering  
   - One-hot encoding of categorical variables  
   - Removal of irrelevant columns using Random Forest Classifier to identify the most important features

4. Model Building  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Gradient Boosting  
   - Artificial Neural Network (ANN)

5. Model Evaluation  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion Matrix

---

## **Results**

The project compared five predictive models to determine the most effective approach for predicting whether employees are likely to seek mental health treatment.

| Model                | Accuracy | F1 Score | Precision | Recall |
|----------------------|----------|----------|-----------|--------|
| Logistic Regression  | 0.7897   | 0.8441   | 0.8102    | 0.8810 |
| Decision Tree        | 0.6513   | 0.7213   | 0.7458    | 0.6984 |
| Random Forest        | 0.7282   | 0.7905   | 0.7874    | 0.7937 |
| Gradient Boosting    | 0.8000   | 0.8494   | 0.8271    | 0.8730 |
| ANN                  | 0.8051   | 0.8468   | 0.8607    | 0.8333 |

### Confusion Matrix Analysis
To better understand model behavior beyond accuracy, confusion matrices were analyzed for all models.

#### Logistic Regression

```text
[[ 43  26]
 [ 15 111]]
```

- Lowest false negatives among all tested learning models.
- Strong recall (**88.1%**) means the model successfully identified most individuals likely to seek treatment.
- Moderate false positives indicate some tendency toward over-predicting positive cases.

#### Decision Tree

```text
[[39 30]
 [39 87]]
```

- Highest number of false negatives among all models.
- Weakest overall performance across all metrics.
- Indicates poor generalization and inability to capture complex feature relationships.

#### Random Forest

```text
[[ 41  28]
 [ 25 101]]
```

- Improved performance compared to Decision Tree due to ensemble learning.
- Better balance between false positives and false negatives.
- More stable predictions than a single decision tree.

#### Gradient Boosting

```text
[[ 46  23]
 [ 16 110]]
```

- Strong overall balance across all error categories.
- Very low false negatives, making it highly effective at identifying employees likely to seek mental health support.
- Best F1 Score (**84.9%**) among all models.

### Artificial Neural Network (ANN)

```text
[[ 52  17]
 [ 21 105]]
```

- Highest accuracy (**80.5%**) among all tested models.
- Highest precision (**86.1%**) indicating fewer false positives than other models.
- Produced the highest true negatives (**52**) showing stronger ability to correctly identify employees unlikely to seek treatment.
- Slightly lower recall than Logistic Regression and Gradient Boosting, meaning more positive cases were missed compared to those models.

### Interpretation of Results

The results demonstrate that both traditional machine learning models and deep learning models can effectively predict treatment-seeking behavior related to mental health.

Several patterns emerged:

- ANN achieved the highest overall accuracy (80.5%), suggesting strong general predictive capability.
- Gradient Boosting achieved the highest F1 Score (84.9%), indicating the best balance between precision and recall.
- Logistic Regression performed exceptionally well despite being the simplest model, demonstrating that simpler linear models can remain highly competitive.
- Decision Tree consistently underperformed, suggesting that individual tree-based approaches are insufficient for this dataset structure.
- Random Forest improved performance over Decision Tree but did not outperform Gradient Boosting, reinforcing the advantage of boosting methods for structured tabular data.


#### Which Model Performed Best?

The answer depends on the objective.

If overall accuracy is the priority, Artificial Neural Network (ANN). This is because ANN performed best in accuracy and precision. it possesses the lowest false positive rate and as a result, is best when reducing unnecessary interventions is important.

If identifying as many positive cases as possible is the priority, Gradient Boosting. Its low false negatives makes it best when minimizing missed at-risk individuals is important.

If interpretability is important, the best model is Logistic Regression. It possesses strong performance despite simplicity and high recall (**88.1%**) which makes it easy to explain feature relationships.

---

### Class Imbalance Handling (SMOTE Experiment)

To address class imbalance in the target variable (`treatment`), the Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training dataset.

This approach was used to generate synthetic samples for the minority class in order to improve model learning and reduce bias toward the majority class.

#### Results After Applying SMOTE

| Model                | Accuracy | F1 Score | Precision | Recall |
|----------------------|----------|----------|-----------|--------|
| Logistic Regression  | 0.8000   | 0.8434   | 0.8537    | 0.8333 |
| Decision Tree        | 0.6667   | 0.7325   | 0.7607    | 0.7063 |
| Random Forest        | 0.7231   | 0.7840   | 0.7903    | 0.7778 |
| Gradient Boosting    | 0.8000   | 0.8421   | 0.8595    | 0.8254 |
| ANN                  | 0.6769   | 0.7097   | 0.8462    | 0.6111 |


#### Key Observations

- SMOTE improved performance for several traditional machine learning models, particularly Logistic Regression and Random Forest, leading to a more balanced precision-recall trade-off.
- Gradient Boosting remained stable, showing minimal sensitivity to class balancing.
- The Artificial Neural Network (ANN) experienced a significant drop in performance, particularly in recall, indicating reduced ability to correctly identify positive cases after synthetic oversampling.


#### Interpretation

The results suggest that the effectiveness of SMOTE is model-dependent:

- Linear and ensemble-based models benefited from synthetic oversampling due to reduced class bias.
- Neural networks were negatively impacted, likely due to the introduction of synthetic samples that may not fully reflect the underlying data distribution in a small tabular dataset.


#### Conclusion

While SMOTE is a useful technique for handling class imbalance, it does not universally improve performance across all model types. In this project, it improved some traditional machine learning models but degraded deep learning performance.

This highlights the importance of evaluating multiple imbalance-handling strategies rather than relying on a single approach.

---

### Key Insights

1. Model Complexity Does Not Guarantee Better Performance

Although ANN performed best in accuracy, the difference between ANN and Gradient Boosting was marginal. This demonstrates that advanced deep learning architectures do not automatically outperform traditional machine learning models.

2. Ensemble Learning Remains Extremely Powerful

Gradient Boosting consistently performed strongly across multiple metrics.

This reinforces the effectiveness of boosting-based algorithms for structured tabular datasets.

3. Accuracy Alone Is Not Enough

ANN achieved the highest accuracy, but Gradient Boosting achieved better F1-score balance. Choosing a model should depend on business objectives rather than accuracy alone.

4. Mental Health Support-Seeking Behavior Is Predictable

The models demonstrated that workplace and behavioral factors contain meaningful predictive patterns associated with treatment-seeking behavior. This suggests predictive analytics can contribute to workplace mental health intervention systems.


### Key Learnings

This project reinforced several important lessons.

#### Technical Learnings

- Proper preprocessing significantly impacts model performance.
- Neural networks require numeric tensor-compatible data types and careful handling of boolean features.
- Early stopping and learning rate tuning greatly improve ANN stability.
- Feature engineering can influence results more than algorithm choice.

#### Machine Learning Learnings

- Traditional machine learning models remain highly competitive.
- Deep learning is not always superior for small tabular datasets.
- Multiple evaluation metrics provide better insight than accuracy alone.

#### Domain Learnings

- Workplace mental health behavior is influenced by multiple demographic and organizational factors.
- Predictive analytics can potentially support early intervention systems.
- Healthcare-related prediction models require careful ethical consideration.

---

### Limitations

Despite strong results, several limitations exist.

#### Geographic Bias (Western Dataset Bias)

The dataset is heavily dominated by respondents from Western countries.

Largest representation:

- United States
- United Kingdom
- Canada
- Germany

African representation was extremely limited.

Examples:

- Nigeria: 1 respondent
- South Africa: 6 respondents

This limits direct applicability to African workplace environments.


#### Small Dataset Size

The dataset contains approximately **1,259 observations**. This creates several challenges:

- Limited generalization
- Higher overfitting risk
- Reduced deep learning robustness


#### Survey-Based Self-Reported Data

The dataset depends entirely on self-reported responses. Possible issues include:

- Response bias
- Social desirability bias
- Inaccurate self-reporting


#### Limited Target Variable Scope

The target variable predicts whether an individual sought treatment. It does not directly predict mental health conditions such as:

- Depression
- Anxiety
- Burnout

Treatment-seeking behavior is also influenced by:

- financial access
- cultural stigma
- workplace culture
- healthcare accessibility

---

### Recommendations for Future Work

- Collecting datasets with stronger African representation
- Using larger datasets for improved deep learning performance
- Testing additional models such as LightGBM
- Building a Streamlit deployment for real-time prediction
- Incorporating fairness and bias evaluation metrics
- Using datasets that directly predict mental health conditions instead of treatment-seeking behavior


### Real-world Application

This model could support Public Health organisations, HR and workplace wellness teams by:

- Identifying employees at risk of not seeking mental health support
- Informing targeted awareness campaigns
- Supporting early intervention strategies
- Improving resource allocation for employee wellbeing programs

### Final Model Ranking

Based on overall performance:

1. Artificial Neural Network (Best accuracy + precision)
2. Gradient Boosting (Best balance: F1 + Recall)
3. Logistic Regression (Best interpretability)
4. Random Forest (Stable baseline model)
5. Decision Tree (Weakest performer)
