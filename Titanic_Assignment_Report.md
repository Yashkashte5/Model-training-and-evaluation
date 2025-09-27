# Titanic Survival Prediction Report

## 1. Dataset and Objective
The dataset used is the Titanic Survival Dataset (publicly available). The objective is to predict whether a passenger survived the Titanic disaster based on features such as age, sex, class, fare, and embarkation port.

---

## 2. Data Understanding and Preprocessing

### Steps Performed
- **Missing values**: Filled missing `age` values with the median; handled missing `embarked` values via mode imputation or drop when appropriate.  
- **Categorical encoding**: Converted categorical variables into appropriate encodings (one-hot encoding for multi-class features, binary encoding for booleans).  
- **Feature engineering**: Added or retained binary flags such as `adult_male` and `alone`; converted `class` to dummy columns (`class_Second`, `class_Third`) where relevant.  
- **Scaling**: Standardized numerical features (`age`, `fare`, `sibsp`, `parch`) using `StandardScaler`.  

Result: A clean dataset with consistent column names matching the model input scheme, ready for modeling.

---

## 3. Exploratory Data Analysis (EDA)

### Key Insights
- **Gender**: Females had a substantially higher survival rate than males.  
- **Passenger class**: First-class passengers had higher survival rates than second and third class.  
- **Age**: Younger passengers (children and young adults) tended to survive more frequently.  
- **Fare**: Higher ticket fares correlated with higher survival probability (proxy for socio-economic status).

### Visualizations (included in the notebook)
1. Histogram of age distribution (by survival).  
2. Scatter plot of age vs fare colored by survival.  
3. Correlation heatmap for numerical features and encoded categorical flags.

---

## 4. Model Building and Evaluation

### Models Trained
- Logistic Regression (baseline).  
- Random Forest Classifier (primary model).

### Evaluation Metrics Used
- Accuracy, Precision, Recall, F1-score, Confusion Matrix.

### Random Forest (Baseline) — Test set results
- **Accuracy**: 0.838  
- **Precision**: 0.803  
- **Recall**: 0.768  
- **F1 Score**: 0.785  

(Full classification report and confusion matrix are available in the notebook.)

---

## 5. Optimization

### Hyperparameter Tuning
- Approach: `GridSearchCV` on the Random Forest classifier.  
- Typical parameter grid used:  
  - `n_estimators`: [100, 200, 300]  
  - `max_depth`: [None, 5, 10, 15]  
  - `min_samples_split`: [2, 5, 10]  
  - `min_samples_leaf`: [1, 2, 4]  
  - `max_features`: ['auto', 'sqrt', 'log2']

- Scoring: F1-score (balanced consideration of precision and recall) was used as the main optimization metric in GridSearchCV for the runs reported here.

### Results — Baseline vs Tuned (Random Forest)
| Model (Random Forest)      | Accuracy | Precision | Recall | F1-score |
|---------------------------:|:--------:|:---------:|:------:|:--------:|
| Baseline (before tuning)   | 0.838    | 0.803     | 0.768  | 0.785    |
| Tuned (after GridSearchCV) | 0.821    | 0.825     | 0.681  | 0.746    |

### Interpretation
- The tuned model **increased precision** (0.803 → 0.825) but **reduced recall** (0.768 → 0.681) and **reduced overall accuracy** (0.838 → 0.821). The F1-score also decreased slightly (0.785 → 0.746).  
- This outcome is expected when the optimization shifts decision boundaries or when the chosen tuning objective prioritizes one trade-off over another. In our case, although GridSearchCV used F1 as the scoring metric, the selected hyperparameters produced a model with higher precision but lower recall. This can happen because the search space and cross-validation splits led to a model that better avoids false positives at the expense of more false negatives.

### Practical implications and recommendations
- **Choose metrics according to the use case**: If false negatives are more costly (i.e., missing survivors is worse), prioritize **recall** or use `class_weight='balanced'` and optimize for recall. If false positives are more costly, prioritize precision.  
- **Threshold calibration**: Rather than relying solely on `predict()`, use `predict_proba()` and choose a probability threshold that gives the desired precision/recall trade-off.  
- **Further tuning**: Consider `RandomizedSearchCV` for a wider parameter search, more cross-validation folds, or expanding the parameter ranges.  
- **Ensembles and stacking**: Combining models (e.g., Logistic Regression + Random Forest) or stacking can improve robustness.  
- **Class weighting / resampling**: If class imbalance affects recall, consider `class_weight='balanced'`, oversampling minority class (SMOTE), or undersampling majority class.  
- **Feature engineering**: Additional features or interactions might improve model expressiveness and overall performance.

---

## 6. Deployment (Streamlit)
- A Streamlit app was built that accepts raw inputs (`pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`) and uses a fitted pipeline (preprocessing + classifier) saved as `titanic_pipeline.pkl` to return a survival prediction and probability. The app and the fitted pipeline are included in the project folder.

---

## 7. Conclusion
- Key predictors for survival were gender, passenger class, and fare.  
- The Random Forest classifier performed strongly. Hyperparameter tuning changed the precision/recall balance and reduced accuracy in this run, highlighting the importance of selecting optimization metrics and further tuning for your specific objective.  
- Deliverables include the Jupyter notebook with code and visualizations, this Markdown report, and the Streamlit app code along with the saved pipeline.

---
