# LLM-Classification-Finetuning
# Chatbot Arena — LLM Preference Prediction

> Kaggle Competition | Class Assignment | Machine Learning

Predicting which LLM response a user will prefer in a head-to-head chatbot battle, using feature engineering and XGBoost classification.

**Competition Link:** https://www.kaggle.com/competitions/lmsys-chatbot-arena

---

## 📌 Problem Statement

Large Language Models (LLMs) generate responses differently — and users don't always prefer the "smarter" one. This competition uses real data from [Chatbot Arena](https://chat.lmsys.org/), where users chat with two anonymous LLMs and vote for the one they prefer.

**Task:** Given a prompt and two responses (`response_a`, `response_b`), predict the probability that:
- `winner_model_a` — users preferred Response A
- `winner_model_b` — users preferred Response B
- `winner_tie` — users found them equally good

**Evaluation Metric:** Multi-class Log Loss (lower is better). Uniform baseline = **1.099**

---

## 📁 Repository Structure

```
├── llm_preference_solution.ipynb   # Main Kaggle notebook (submit this)
├── llm_preference_solution.py      # Same solution as a plain Python script
└── README.md                       # You are here
```

---

## 🧠 Approach

### 1. Feature Engineering
Extracted **12 quality-signal features** from each response:

| Feature | Description |
|---|---|
| `word_count` | Total words in response |
| `char_len` | Total character length |
| `has_code` | Contains code block (` ``` `) |
| `numbered_steps` | Count of numbered list items |
| `bullet_count` | Count of bullet points |
| `header_count` | Count of markdown headers |
| `avg_word_len` | Average word length |
| `avg_sent_len` | Average sentence length |
| `has_examples` | Mentions "for example", "e.g." etc. |
| `has_sources` | Mentions sources/references |
| `starts_direct` | Does NOT start with "Sure!", "Certainly!" etc. |
| `hedge_count` | Count of uncertain words (might, could, perhaps) |

For each row, three sets of features are created: **per-response** (`a_*`, `b_*`), **difference** (`diff_*`), and **ratio** (`ratio_*`) — giving 36 total features per training sample.

### 2. Model
- **XGBoost** multiclass classifier (3 classes: model_a, model_b, tie)
- 300 estimators, learning rate 0.05, max depth 5
- Falls back to **heuristic softmax scoring** if training data is unavailable

### 3. Label Handling
The notebook auto-detects both train.csv formats:
- Single `winner` column (`model_a` / `model_b` / `tie`)
- One-hot columns (`winner_model_a`, `winner_model_b`, `winner_tie`)

---

## 🚀 How to Run

### On Kaggle (for submission)
1. Open `llm_preference_solution.ipynb` in a Kaggle Notebook
2. Set the dataset to `lmsys-chatbot-arena`
3. Click **Save Version → Save & Run All**
4. Once complete, go to **Output** tab and click **Submit to Competition**

### Locally
```bash
# Install dependencies
pip install pandas numpy xgboost

# Run the script
python llm_preference_solution.py
```
> Update `INPUT_DIR` in the CONFIG section to point to your local data folder.

---

## 📊 Results

| Model | Log Loss |
|---|---|
| Uniform baseline | 1.0986 |
| Heuristic scoring | ~1.05 (estimated) |
| XGBoost on quality features | TBD after submission |

---

## 💡 What I Learned

- How to frame a **preference prediction** problem as multiclass classification
- The importance of **feature engineering** on text data without using embeddings
- How Kaggle's **code competition** submission pipeline works (notebook re-run with hidden test set)
- Handling **inconsistent dataset schemas** (two different label formats in the same competition)
- How **RLHF reward models** work conceptually — this task is essentially building one from scratch

---

## 🔮 Future Improvements

- Use **TF-IDF** or **sentence embeddings** (e.g. SBERT) as additional features
- Use a pre-trained LLM (e.g. DeBERTa) to directly score response quality
- Add **prompt-aware features** — some prompt types reward different response styles
- Tune the **tie probability prior** based on training data distribution
- Cross-validation to prevent overfitting

---

## 🛠️ Tech Stack

- Python 3.10+
- pandas, numpy
- XGBoost
- Kaggle Notebooks (GPU/CPU)

---

## 📚 References

- [Chatbot Arena Paper — LMSYS](https://arxiv.org/abs/2306.05685)
- [RLHF — Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
