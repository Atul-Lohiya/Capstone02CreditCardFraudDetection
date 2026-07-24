# """
# Section 1 - Imports, artifact loading, metadata extraction and helper utilities.
# Append Sections 2 and 3 below this in the same app.py.
# """

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

ARTIFACT_PATH = "credit_card_fraud_detector.joblib"

@st.cache_resource
def load_artifact():
    artifact = joblib.load(ARTIFACT_PATH)
    return artifact

artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_order = artifact["features"]
metadata = artifact.get("feature_metadata", {}) # this has stats like mean, median, quratiles etc for all, non-fraud and fraud classes
metadata_all = metadata["all"]
metadata_non_fraud = metadata["non-fraud"]
metadata_fraud = metadata["fraud"]
# feature_order = metadata.keys()

def feature_info(feature, sample_class):
    meta = metadata_all

    if sample_class == 0:
        meta = metadata_non_fraud
    elif sample_class == 1:
        meta = metadata_fraud

    # return metadata.get(feature, {})
    return meta.get(feature, {})

def random_value(feature, sample_class):
    
    info = feature_info(feature, sample_class)
    median = info.get("median")
    mn = info.get("min")
    mx = info.get("max")
    iqr = info.get("iqr")
    sigma = iqr / 1.349 # standard formula for IQR = 1.349 * sigma
    value = float(np.random.normal(median, sigma))
    # return np.clip(
    #     value,
    #     mn,
    #     mx)   # clip if you need to constrain it between min and max for the training dataset, if not return the unbounded value
    return value

def random_record():
    choice = random.choices(
                population=[-1, 0, 1],
                weights=[0.375, 0.375, 0.25],
                k=1
            )[0]
    record = {}
    for f in feature_order:
        value = random_value(f, choice)
        if f.lower() == "amount" or f.lower() == "time":
            value = max(0.0, value)
        record[f] = value
        record["class"] = choice
    return record

def records_to_dataframe(records):
    df = pd.DataFrame(records)
    return df[feature_order]

def predict_records(records):
    df = records_to_dataframe(records)
    pred = pipeline.predict(df)
    prob = pipeline.predict_proba(df)[:,1]
    out = df.copy()
    out["Prediction"] = pred
    out["Fraud Probability"] = prob
    return out


# """
# Section 2 - Streamlit UI.
# Append below Section 1.
# """

st.title("Credit Card Fraud Detection")

# =============================================================================
# Information about Synthetic Data Generation
# =============================================================================

with st.expander("About Synthetic Sample Generation", expanded=False):

    st.markdown("""
### Why synthetic sample generation?

This application provides automatic sample generation to make it easier to
demonstrate model inference.

The dataset contains **28 PCA-transformed features (V1–V28)** whose original
business meanings are intentionally unavailable. Because these principal
components do not have directly interpretable ranges, manually creating
realistic transactions can be difficult.

Synthetic transaction generation helps users quickly test the deployed model.
The generated values may also be manually edited before prediction.

---

### How are random transactions generated?

Three statistical profiles are derived from the training dataset:

- **Overall dataset (class=-1)**
- **Non-Fraud transactions (class=0)**
- **Fraud transactions (class=1)**

Whenever a transaction is generated, one of these profiles is randomly selected
using weighted sampling.

Each feature is then generated independently by sampling from a **Normal
distribution** centered on the feature's **median**, with the spread estimated
from the feature's **Interquartile Range (IQR)**.

For the **TIME** and **AMOUNT** features, negative values are prevented since
they are not meaningful.

---

### Important note

The generated transactions are **synthetic** and are intended only for
demonstrating model inference.

Feature values are generated independently and therefore **do not preserve the
true relationships (correlations)** that exist among variables in real credit
card transactions. Consequently, these samples should **not** be interpreted as
real transactions or used for model evaluation.

Model performance reported in this project is based exclusively on the original
training and test datasets.
""")

    st.subheader("Training Data Statistics")

    tab1, tab2, tab3 = st.tabs([
        "Overall (class=-1)",
        "Non-Fraud (class=0)",
        "Fraud (class=1)"
    ])

    with tab1:
        st.dataframe(
            pd.DataFrame(metadata_all).T,
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            pd.DataFrame(metadata_non_fraud).T,
            use_container_width=True
        )

    with tab3:
        st.dataframe(
            pd.DataFrame(metadata_fraud).T,
            use_container_width=True
        )

st.info(
"""TIME is seconds since the first transaction in the dataset.
V1-V28 are PCA components. You can either fill them manually or click "Auto-fill fields" to aut-fill them based on deployed training metadata like quartiles, min, max etc.
Deployment uses the default classification threshold of 0.5, matching model evaluation."""
)

num_cols = 5
cols=st.columns(num_cols)
class_mapping = {
    -1: "Overall Training Statistics (class=-1)",
     0: "Non-Fraud Statistics (class=0)",
     1: "Fraud Statistics (class=1)"
}

for i,f in enumerate(feature_order):
    col=cols[i % num_cols]
    # info=feature_info(f, -1)
    with col:
        st.number_input(
            f,
            # value=float(info.get("median",0.0)),  # set median as default
            value=float(0.0),
            # min_value=float(info.get("min",-1e9)),
            # max_value=float(info.get("max",1e9)),
            step=0.01,
            key=f
        )
if "class" in st.session_state:
    st.info(
        f"Random sample generated using **{st.session_state['class']}**."
    )

def autofill_fields():
    record = random_record()

    for k, v in record.items():
        # if k == "class": # skip class property as that is not a user field
        #     continue
        st.session_state[k] = class_mapping[v] if k == "class" else v

buttons_row = st.columns([1, 1, 10], gap="small")
with buttons_row[0]:
    autofill = st.button("Auto-fill", on_click=autofill_fields)

with buttons_row[1]:
    predict_single = st.button("Predict")

with buttons_row[2]:
    st.html("""
    <div style="
        display:flex;
        align-items:center;
        height:38px;
        font-size:14px;
        color:#666;
    ">Scroll to bottom for results</div>
    """)

st.divider()
st.subheader("Batch prediction (JSON)")
st.write("Auto-generate records")
buttons_row = st.columns([1.5, 1.5, 9], gap="small")

with buttons_row[0]:
    count=st.number_input("Auto-generate records",1,100,50, label_visibility="collapsed")
with buttons_row[1]:
    gen_json=st.button("Generate JSON")
with buttons_row[2]:
    predict_json=st.button("Predict JSON")

if gen_json:
    st.session_state["json_text"]=json.dumps(
        [random_record() for _ in range(int(count))],
        indent=2
    )

json_text=st.text_area(
    "JSON array of transaction objects",
    value=st.session_state.get("json_text",""),
    height=350
)



# """
# Section 3 - Inference, Results and Main App

# Paste below Sections 1 and 2.
# """

# ---- Individual prediction ----
if predict_single:
    rec = {k: st.session_state[k] for k in feature_order}
    # rec = st.session_state["single_record"]
    df = records_to_dataframe([rec])
    probs = pipeline.predict_proba(df)[:,1]
    preds = pipeline.predict(df)
    out = df.copy()
    out["Prediction"]=preds
    out["Fraud Probability"]=probs
    st.dataframe(out, use_container_width=True)

# ---- Batch prediction ----
if predict_json:
    try:
        records=json.loads(json_text)
        if not isinstance(records,list):
            raise ValueError("JSON must be a list of objects.")
        df=records_to_dataframe(records)
        probs=pipeline.predict_proba(df)[:,1]
        preds=pipeline.predict(df)
        out=df.copy()
        out["Prediction"]=preds
        out["Fraud Probability"]=probs
        st.dataframe(out,use_container_width=True)
    except Exception as e:
        st.error(str(e))

