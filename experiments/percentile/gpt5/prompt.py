PROMPT_TEMPLATE = """## Task

Analyze code solutions for the Kaggle competition **{task_name}** by first documenting what each solution actually implements, then identifying patterns through comparison.

---

## Competition Context

<task_info>
{task_info}
</task_info>

---

## Part 0: Task Analysis

Before analyzing solutions, extract key information about the competition task:

#### 1. Task Characterization
- **Problem type:** [Classification, Regression, NLP, Computer Vision, Time Series, Recommendation, Multi-modal, etc.]
- **Specific task:** [Binary classification, multi-class, object detection, sequence prediction, etc.]
- **Domain:** [Healthcare, finance, e-commerce, social media, etc.]
- **Input data format:** [Tabular, images, text, audio, mixed, etc.]
- **Output format:** [Single prediction, multiple predictions, probability distributions, etc.]

#### 2. Evaluation Metric
- **Primary metric:** [Accuracy, F1, AUC-ROC, RMSE, MAE, etc.]
- **Metric implications:**
  - What does this metric optimize for? (e.g., balanced classes, ranking, absolute error)
  - Does it penalize certain errors more than others?
  - How does it affect model selection and training strategy?
- **Submission format requirements:** [Probability scores, class labels, rankings, etc.]

#### 3. Task Constraints & Requirements
- **Data constraints:**
  - Dataset size (number of samples, features)
  - Class imbalance (if applicable)
  - Missing data prevalence
  - Data quality issues mentioned

#### 4. Task Difficulty Analysis
**What makes this task challenging?**
- **Data challenges:** [Imbalance, noise, missing values, high dimensionality, small sample size, etc.]
- **Modeling challenges:** [Complex patterns, need for domain knowledge, feature engineering difficulty, etc.]
- **Evaluation challenges:** [Metric sensitivity, overfitting risks, validation strategy complexity, etc.]
- **Domain-specific challenges:** [Any unique aspects of this domain/problem]

---

## Part 1: Individual Solution Summaries

For EACH solution, extract and document the following factual information:

### Solution Summary Template

**Solution ID:** [identifier]  
**Score Percentile:** [percentile]

#### 1. Data Preprocessing
- Input data loading method
- Missing value handling (method, columns affected)
- Data cleaning steps (outlier removal, filtering, etc.)
- Normalization/scaling (which columns, which method)
- Data type conversions
- Train/test split approach
- Other preprocessing steps: [Any other data preparation not covered above]

#### 2. Feature Engineering  
- Features created (list each with formula/method if possible)
- Feature selection/reduction techniques used
- Domain-specific transformations
- Interaction terms or polynomial features
- Time-based features (if applicable)
- Other feature engineering: [Any other feature work not covered above]

#### 3. Synthetic Data / Data Augmentation
- Whether synthetic data was generated: Yes/No
- If yes: Generation method, volume, and integration approach
- Specific augmentation techniques used
- Other data generation approaches: [Any other synthetic/augmentation methods]

#### 4. Model Selection
- Primary algorithm(s) used (exact model class/function)
- Model hyperparameters (learning rate, depth, n_estimators, etc.)
- Ensemble approach (if any): stacking, blending, voting, etc.
- Number of models in ensemble
- Pretrained models: [Which models, from where (ImageNet, HuggingFace, etc.)]
- External datasets: [Any additional data used, sources]
- Other architectural details: [Loss functions, custom layers, regularization, etc.]

#### 5. Training Methodology
- Cross-validation scheme (k-fold, stratified, time-series split, etc.)
- Hyperparameter tuning approach (grid search, random search, Bayesian, manual)
- Training/validation split ratios
- Early stopping criteria (if applicable)
- Number of training epochs/iterations
- Other training details: [Learning rate schedules, callbacks, optimization tricks, etc.]

#### 6. Evaluation & Submission
- Final prediction method (mean, median, weighted average, etc.)
- Post-processing of predictions
- Other evaluation/submission details: [Threshold tuning, calibration, etc.]

#### 7. Notable Implementation Details
[Catch-all for anything important not captured above]
- [Any unique approaches, novel techniques, or important details]
- [Computational considerations (GPU usage, runtime optimizations)]
- [Anything else that seems significant to the solution's approach]

---

## Part 2: Comparative Analysis

After documenting all solutions above, provide:


### A. Solution Classification

First, identify and categorize all solutions by performance:

**High Score Solutions (percentile_score >= 0.6):**
- Solution ID: [ID], Score: [percentile_score]
- Solution ID: [ID], Score: [percentile_score]
- ...
Total: [N] high-score solutions

**Low Score Solutions (percentile_score < 0.6):**
- Solution ID: [ID], Score: [percentile_score]
- Solution ID: [ID], Score: [percentile_score]
- ...
Total: [N] low-score solutions

### B. Pattern Matrix

Create a table comparing key implementation choices:

| Dimension | High Score Implementations | Low Score Implementations |
|-----------|---------------------------|---------------------------|
| Data preprocessing | [List methods used across high-score solutions] | [List methods used across low-score solutions] |
| Feature engineering | ... | ... |
| Synthetic data | ... | ... |
| Model selection | ... | ... |
| Training methodology | ... | ... |
| Evaluation & submission | ... | ... |
| Notable details | ... | ... |

### C. Critical Differences

For each dimension where high and low scores diverge significantly:

**[Dimension name]**
- **What high-score solutions did:** [Factual description with frequency: "3/4 used..."]
- **What low-score solutions did:** [Factual description with frequency]
- **Concrete difference:** [Specific technical difference]
- **Observed correlation:** [Note: not causation, just correlation]

### D. Consistency Analysis

- **High-score convergence:** Which techniques appeared frequently appeared in (>=50%) high-score solutions?
- **Low-score anti-patterns:** Which mistakes/omissions appeared in MOST (>=50%) low-score solutions?
- **High-score diversity:** Where did successful solutions differ from each other?

---

## Part 3: Solution Space Exploration

**Understanding potential approach diversity:**
- **Multiple valid approaches:** Are there fundamentally different ways to solve this task?
  - [e.g., "Tree-based vs neural network approaches"]
  - [e.g., "Feature engineering-heavy vs end-to-end learning"]
  - [e.g., "Simple model + complex features vs complex model + simple features"]
- **Expected diversity dimensions:**
  - Where might successful solutions legitimately differ?
  - Which choices are likely task-critical vs. flexible?
- **Ensemble potential:** Could diverse approaches be complementary?

---

## Output Guidelines

- Be **factual and specific**: Reference actual code, not general principles
- **Quantify patterns**: Use frequencies (e.g., "3 out of 4 solutions")
- **Quote or cite code** when describing techniques
- **Distinguish correlation from causation**: Avoid claiming X causes Y without evidence
- **Note ambiguities**: If solutions are too diverse to find clear patterns, say so
- **Flag missing information**: If a solution doesn't clearly implement something, note it explicitly
- **Connect back to task analysis**: When identifying patterns, relate them to the task characteristics and challenges"""