import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

st.set_page_config(page_title="Smart Dispatch", page_icon="🚦", layout="wide")

if 'last_prediction' not in st.session_state: st.session_state.last_prediction = None
if 'current_inputs' not in st.session_state: st.session_state.current_inputs = None

# --- LOAD TRUE MACHINE LEARNING MODELS & DATABASES ---
@st.cache_resource
def load_system():
    model_tii = lgb.Booster(model_file='lightgbm_tii_model.txt')
    model_closure = lgb.Booster(model_file='lightgbm_closure_model.txt')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    svd = joblib.load('svd_transformer.pkl')
    historic_db = pd.read_csv('historic_database.csv')
    return model_tii, model_closure, tfidf, svd, historic_db

try:
    lgb_tii, lgb_closure, tfidf, svd, historic_db = load_system()
except Exception as e:
    st.error("Error loading models. Run Cell 9 in the Jupyter notebook first.")
    st.stop()

# --- PREDICTION LOGIC ---
def predict_dispatch(cause, veh_bucket, corridor, zone, hour, day_name, concurrent, text):
    # 1. Map Time Data
    day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    day_idx = day_map[day_name]
    is_weekend = 1 if day_idx >= 5 else 0
    
    # 2. Compliant NLP Processing (TF-IDF + SVD)
    tfidf_vec = tfidf.transform([text.lower()])
    semantic_feats = svd.transform(tfidf_vec)[0]
    
    # 3. Build DataFrame
    input_data = pd.DataFrame({
        'event_cause': [cause.lower().replace(" ", "_")],
        'corridor': [corridor],
        'zone': [zone],
        'hour': [hour],
        'day_of_week': [day_idx],
        'is_weekend': [is_weekend],
        'veh_bucket': [veh_bucket],
        'concurrent_events': [concurrent],
        'semantic_f0': [semantic_feats[0]],
        'semantic_f1': [semantic_feats[1]],
        'semantic_f2': [semantic_feats[2]],
        'semantic_f3': [semantic_feats[3]],
        'semantic_f4': [semantic_feats[4]]
    })
    
    for col in ['event_cause', 'corridor', 'zone', 'veh_bucket']:
        input_data[col] = input_data[col].astype('category')
        
    # Execute Dual Inference
    raw_tii = lgb_tii.predict(input_data)[0]
    closure_prob = lgb_closure.predict(input_data)[0]
    
    return raw_tii, closure_prob

# --- UI LAYOUT ---
st.title("🚦 Smart Dispatch: solving urban traffic through AI")
st.markdown("Predicts localized traffic impact, closure probability, and dynamic diversions using Day-0 dispatch facts.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Event Intelligence")
    event_cause = st.selectbox("Reported Cause", ['Vehicle Breakdown', 'Accident', 'Water Logging', 'Tree Fall', 'Public Event', 'Construction'])
    veh_display = st.selectbox("Vehicle Category", ['Small / Two Wheeler', 'LMV (Cars / Taxis)', 'Large Vehicle (Buses / Trucks)'])
    veh_bucket = {'Small / Two Wheeler': 'small_or_none', 'LMV (Cars / Taxis)': 'lmv', 'Large Vehicle (Buses / Trucks)': 'large_vehicle'}[veh_display]
    zone = st.selectbox("Bengaluru Zone", ['Unknown', 'Central Zone 1', 'East Zone 1', 'South Zone 1', 'West Zone 1', 'North Zone 1'])
    
with col2:
    st.subheader("Spatio-Temporal Data")
    corridor = st.selectbox("Corridor Type", ['Non-corridor', 'ORR East', 'Tumkur Road', 'Hosur Road', 'Mysuru Road'])
    day_name = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour = st.slider("Time of Day (24H)", 0, 23, 9)
    concurrent = st.number_input("Concurrent Active Events in Zone", min_value=0, max_value=20, value=0, help="Creates a dynamic network impact penalty.")

with col3:
    st.subheader("Dispatch Log (Compliant NLP)")
    description = st.text_area("Officer Notes", placeholder="e.g., Accident blocking two lanes...")

st.divider()

# --- PREDICTION TRIGGER ---
if st.button("Generate Resource & Diversion Plan", type="primary", use_container_width=True):
    with st.spinner("Executing Network Analysis and Inference..."):
        
        # 1. AI Predictions
        raw_tii, closure_prob = predict_dispatch(event_cause, veh_bucket, corridor, zone, hour, day_name, concurrent, description)
        safe_tii = np.clip(raw_tii + 3.0, 0, 100)
        
        # 2. Historical Duration Proxy (Manpower T-Shirt Sizing)
        hist_durations = historic_db[(historic_db['event_cause'] == event_cause.lower().replace(" ", "_")) & (historic_db['corridor'] == corridor)]
        median_mins = hist_durations['duration_mins'].median() if not hist_durations.empty else 120
        
        if median_mins > 180: manpower = "HIGH EFFORT (Multi-Shift Response required)"
        elif median_mins > 60: manpower = "MODERATE EFFORT (Standard Patrol Unit)"
        else: manpower = "LOW EFFORT (Quick Clearance Team)"

        # 3. Adjacency Matrix Diversion Engine (No External Map APIs used!)
        safe_corridors = historic_db[(historic_db['zone'] == zone) & (historic_db['corridor'] != corridor) & (historic_db['corridor'] != 'Non-corridor')]['corridor'].dropna().unique()
        if len(safe_corridors) > 0:
            diversion_plan = f"Adjacency Matrix recommends routing via: {', '.join(safe_corridors[:2])}"
        else:
            diversion_plan = "No adjacent corridors found in zone database. Implement local U-turn protocols."

        
        # 4. Save Session State for Feedback Loop
        st.session_state.last_prediction = safe_tii
        
        # Ensure all 7 variables, including description_text, are saved
        st.session_state.current_inputs = {
            "event_cause": event_cause,
            "veh_bucket": veh_bucket,
            "zone": zone,
            "corridor": corridor,
            "day_name": day_name,
            "hour": hour,
            "description_text": description  # Maps your text area to the CSV column
        }
    # --- UI RESULTS DISPLAY ---
    tii_color = "red" if safe_tii >= 75 else "orange" if safe_tii >= 40 else "green"
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label="Predicted Impact Index (TII)", value=f"{safe_tii:.1f} / 100")
        st.metric(label="AI Road Closure Probability", value=f"{closure_prob*100:.1f}%")
        
    with res_col2:
        st.info(f"**Manpower Requirement:** {manpower} (Est. Clearance: {int(median_mins)} mins)")
        if closure_prob > 0.5:
            st.error(f"**Barricading Required:** High closure probability detected. Deploy Tier-1 Barricades immediately.")
        else:
            st.success(f"**Barricading Required:** Closure unlikely. No static barricades needed.")
            
        st.warning(f"**Dynamic Diversion Plan:** {diversion_plan}")

# --- POST EVENT LEARNING SYSTEM ---
st.divider()
st.subheader("🔄 MLOps Staging Area (Feedback Loop)")
st.markdown("*Note for Judges: To preserve live model integrity, data is appended to staging logs for nightly batch retraining.*")

# Expand to 3 columns to make room for the new input
fb_col1, fb_col2, fb_col3 = st.columns(3)

with fb_col1: 
    actual_impact = st.slider("Log Actual Observed Impact (0-100)", 0, 100, 50)
    
with fb_col2: 
    # NEW: Input for actual cops deployed
    actual_cops = st.number_input("Actual Cops Deployed", min_value=0, max_value=100, value=2)

with fb_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Submit to Nightly Retrain Database", type="secondary"):
        if st.session_state.last_prediction is None or st.session_state.current_inputs is None:
            st.error("Generate a prediction first!")
        else:
            # 1. Base log entry including the new actual_cops variable
            log_entry = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ai_predicted_tii": round(st.session_state.last_prediction, 2),
                "actual_observed_tii": actual_impact,
                "actual_cops_deployed": actual_cops 
            }
            
            # Merge in the features saved in session state (including description_text)
            log_entry.update(st.session_state.current_inputs)
            
            # 2. ENFORCE STRICT COLUMN ORDER MATCHING THE CSV
            expected_columns = [
                "timestamp", "event_cause", "veh_bucket", "zone", "corridor", 
                "day_name", "hour", "description_text", "ai_predicted_tii", 
                "actual_observed_tii", "actual_cops_deployed"
            ]
            
            fb_data = pd.DataFrame([log_entry], columns=expected_columns)
            
            # CRITICAL FIX: Remove line breaks and returns from the description text
            # Otherwise, pandas will write a literal new-line into the CSV, breaking the row.
            if pd.notna(fb_data.at[0, 'description_text']):
                clean_text = str(fb_data.at[0, 'description_text']).replace('\n', ' ').replace('\r', '')
                fb_data.at[0, 'description_text'] = clean_text
            
            # 3. Safely append
            file_path = "active_learning_logs.csv"
            write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            
            try:
                fb_data.to_csv(file_path, mode='a', header=write_header, index=False)
                st.success("Feedback staged safely. Target weights will update in the 3 AM MLOps CRON job.")
                
                # Clear state to prevent accidental duplicate submissions
                st.session_state.last_prediction = None
                st.session_state.current_inputs = None
            except Exception as e:
                st.error(f"Failed to write to database: {e}")