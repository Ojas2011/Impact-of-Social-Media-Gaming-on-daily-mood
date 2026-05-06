# AI System to Detect Mental Health Signals from Gaming & Social Media Behavior

**School of Data Science, UNC Charlotte | Spring 2026**

This project applies machine learning to analyze self-reported behavioral patterns from a survey of UNC Charlotte students and the Charlotte, NC community to detect potential mental health signals.

---

## Research Questions

- How does social media & gaming usage affect daily mental health?
- Does increased screen time correlate with worse mood?
- Does gaming reduce or increase stress levels?
- Can AI predict mental health signals from user behavior?

---

## Repository Structure

```
├── data/
│   └── survey_data.csv          # Synthetic survey dataset (70 rows, 11 features)
├── Mental_Health_Gaming_Analysis.ipynb   # Main analysis notebook
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Self-collected Google Form, UNC Charlotte community, April 2026 |
| Size | 100 raw responses → 70 usable rows, 11 features after cleaning |
| Target | Daily Mood — **Bad** (0–1) \| **Neutral** (2) \| **Good** (3–4) |
| Split | 80/20 train/test → 56 training, 14 test samples |

**Features:** Social Media Hours, Video Game Hours, Anxiety Level, Social Comparison, Self Esteem, Screen Use for Relief, Loneliness, Sleep Affected, Addiction Level, Gamer Status

---

## Preprocessing Steps

1. Dropped 27 irrelevant / free-text columns
2. Ordinal encoding for Social Media & Video Game Hours
3. Median Imputation (`SimpleImputer`) for missing values
4. Target binned: Bad (0–1), Neutral (2), Good (3–4)
5. 80/20 train/test split → 56 train, 14 test samples
6. Removed erroneous `'Yes'` value from Social Media Hours

---

## Models

| Model | Description |
|---|---|
| **SVM** | RBF kernel, `class_weight='balanced'`, finds max-margin hyperplane |
| **KNN (k=6)** | Euclidean distance, StandardScaler, k=1–20 tuned, k=6 optimal |
| **Linear Regression** | Baseline: tests continuous mood prediction |
| **Decision Tree** | `max_depth=4`, trained on all features vs. screen-time only |
| **Calibrated SVM** | `CalibratedClassifierCV` wraps SVM via isotonic regression, 5-fold CV |

---

## Results

| Model | Test Accuracy |
|---|---|
| **KNN (k=6)** | **64.3%** |
| Decision Tree (All Features) | 57.1% |
| SVM | 50.0% |
| Calibrated SVM | 50.0% |
| Decision Tree (Screen-Time Only) | 42.9% |
| Linear Regression R² | −0.02 |

### Key Findings

- **KNN (k=6) was the best model** — smallest train/test gap, least overfitting
- **Psychological factors beat screen time** — Anxiety, Addiction, Sleep Disruption, and Loneliness outranked raw screen-time hours as predictors of daily mood
- **Avoidance coping backfires** — using screens to escape stress correlated with lower mood
- **Mood is non-linear** — negative R² confirms mood cannot be predicted linearly from screen time, validating the classifier approach
- **Screen-time only accuracy dropped**: 57.1% → 42.9%

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Mental_Health_Gaming_Analysis.ipynb
```

---

## Limitations & Ethics

- **Small sample** (~70 usable rows, mostly UNCC students) — results may not generalise nationally
- **Self-reported bias** — respondents may misremember screen time or mood
- **Class imbalance** — 'Neutral' class is underrepresented in some splits
- **Correlation ≠ Causation** — academic stress may independently explain both screen time and low mood
- **Privacy** — all responses fully anonymous; no personally identifiable information collected
- **Not clinical** — for awareness and education only; never use for diagnosis or screening

---

## References

- Psychology Today — Benefits of social media for community connection
- Pew Research Center — Teen pressure and social media anxiety
- Scikit-learn — `CalibratedClassifierCV`, KNN, SVC, `DecisionTreeClassifier`

---

*AI Transparency: Claude (Anthropic, claude sonnet 4.6) was used to help fix code errors.*
