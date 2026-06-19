# 🚦 TrafficOps AI: Constraint-Driven Urban Dispatch Intelligence

An edge-ready, dual-AI pipeline built for the **Gridlock Hackathon 2.0**. Smart Dispatch transitions city traffic management from a reactive, experience-driven model to a proactive, data-driven system by instantly predicting the **Traffic Impact Index (TII)** and **Road Closure Probabilities** from Day-0 dispatcher notes.

### 🏆 The Hackathon Challenge & Our Approach
Urban gridlock forms *before* officers arrive on the scene to assess severity. We needed to predict impact and route diversions **without** relying on forbidden external mapping APIs (like Google Maps) or heavy cloud LLMs. 

To solve this, we built a fully localized, sub-30-millisecond inference engine that extracts physical traffic patterns and dynamic diversions strictly from the provided historical dataset.

---

## 🚀 Key Innovations & Features

* **Dual-AI Inference Engine:** Runs two parallel LightGBM models locally. 
    * *Model 1 (Regression):* Predicts the 100-point Traffic Impact Index (TII) to determine response urgency.
    * *Model 2 (Classification):* Predicts the exact probability of a road closure to automate barricade deployment.
* **API-Free Dynamic Diversions:** We bypassed external map APIs and use of external datasets by engineering an internal **Spatial Adjacency Matrix**. The system queries a lightweight historic topology database to recommend safe, adjacent corridors within the same operational zone.
* **Compliant NLP Pipeline:** Bypassed heavy, latency-inducing LLMs. We utilized a native TF-IDF vectorizer paired with Truncated SVD (Latent Semantic Analysis) to instantly convert messy officer notes into dense mathematical arrays.
* **Nightly MLOps Feedback Loop:** To protect against live data-poisoning, human operator corrections are safely appended to an isolated staging log (`active_learning_logs.csv`). A simulated 3:00 AM batch job merges this feedback to organically retrain the model weights.
* **Edge-Ready Latency:** By decoupling the heavy historical training data from the live Streamlit app, the end-to-end composite inference executes in **~26 milliseconds**.

---

## 🧠 System Architecture

Our codebase is strictly divided to ensure production-level speed and safety:

1.  **Track 1: Offline Training (`traffic.ipynb`)**
    * Ingests the raw, sparse dataset.
    * Executes *Synthetic Time Projection* to salvage missing operational timestamps.
    * Calculates concurrent network overlaps (Event-driven congestion).
    * Trains models and exports lightweight `.txt` and `.pkl` artifacts.
2.  **Track 2: Live Edge Inference (`app.py`)**
    * A Streamlit-powered Command Dashboard.
    * Loads pre-compiled artifacts and a minimal `historic_database.csv` dictionary to achieve zero-lag inference.

---

## 📂 Project Structure

```text
├── app.py                      # Live Streamlit dispatcher dashboard
├── traffic.ipynb               # Offline data engineering & model training notebook
├── requirements.txt            # Python dependencies
├── historic_database.csv       # Lightweight topological lookup dictionary
├── active_learning_logs.csv    # Staging area for human-in-the-loop ML feedback
├── lightgbm_tii_model.txt      # Compiled Model 1 (Regression)
├── lightgbm_closure_model.txt  # Compiled Model 2 (Classification)
├── tfidf_vectorizer.pkl        # Compiled NLP Vocabulary
└── svd_transformer.pkl         # Compiled Semantic Feature Extractor
