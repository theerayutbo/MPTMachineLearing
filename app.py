import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- App Config ---
st.set_page_config(page_title="ML GUI: Data Cleaning & Model Training",
                   layout="wide")
st.title("📊 ML GUI: Data Cleaning & Model Training")

import os
import glob

# --- Globals ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = None

# --- Sidebar ---
st.sidebar.header("Data Loading")
data_dir = st.sidebar.text_input("Enter the path to the Measurement_Data directory", "Measurement_Data/")
if st.sidebar.button("Load and Process Data"):
    if not os.path.isdir(data_dir):
        st.error(f"Directory not found: {data_dir}")
        st.stop()

    search_path = os.path.join(data_dir, 'metal', '*', '*', '*', '*', 'summary_results.csv')
    all_files = glob.glob(search_path)
    if not all_files:
        st.error(f"No 'summary_results.csv' files found. Searched in: {search_path}")
        st.stop()

    df_list = []
    for file in all_files:
        try:
            # Extract info from path
            parts = file.split(os.sep)
            metal_type = parts[-5]
            sample = parts[-3]
            direction = parts[-2]

            temp_df = pd.read_csv(file)
            temp_df['metal_type'] = metal_type
            temp_df['sample'] = sample
            temp_df['direction'] = direction
            df_list.append(temp_df)
        except Exception as e:
            st.warning(f"Could not process file {file}: {e}")

    if not df_list:
        st.error("No data could be loaded. Please check the files.")
        st.stop()

    master_df = pd.concat(df_list, ignore_index=True)
    st.session_state.master_df = master_df
    st.success("Data loaded and processed successfully!")

# --- Filtering ---
if st.session_state.master_df is not None:
    st.sidebar.header("Filtering")
    df = st.session_state.master_df

    # Get unique values for filters
    metals = df['metal_type'].unique()
    samples = df['sample'].unique()
    directions = df['direction'].unique()

    # Create filters
    selected_metals = st.sidebar.multiselect("Select Metal Type", metals, default=metals)
    selected_samples = st.sidebar.multiselect("Select Sample", samples, default=samples)
    selected_directions = st.sidebar.multiselect("Select Direction", directions, default=directions)

    # Apply filters
    filtered_df = df[
        df['metal_type'].isin(selected_metals) &
        df['sample'].isin(selected_samples) &
        df['direction'].isin(selected_directions)
    ]
    st.session_state.filtered_df = filtered_df

# --- Tabs ---
tab_load, tab_train = st.tabs(["📤 Load Data", "🤖 Train Model"])

with tab_load:
    st.header("Interactive Plot")
    if 'filtered_df' in st.session_state:
        df = st.session_state.filtered_df

        st.subheader("Data Preview")
        st.dataframe(df)

        st.subheader("Create Plot")
        x_axis = st.selectbox("Select X-axis", df.columns, index=0)
        y_axes = st.multiselect("Select Y-axis", df.columns, default=df.columns[1:3].tolist())

        if x_axis and y_axes:
            fig = go.Figure()
            for y_axis in y_axes:
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='lines+markers', name=y_axis))

            fig.update_layout(
                title=f"{', '.join(y_axes)} vs {x_axis}",
                xaxis_title=x_axis,
                yaxis_title="Value",
                xaxis_type="log"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load data using the sidebar control and apply filters.")

with tab_train:
    st.header('Model Training')
    if st.session_state.master_df is not None:
        data = st.session_state.master_df
        st.subheader('Dataset Preview')
        st.dataframe(data.head(), use_container_width=True)
        all_cols = data.columns.tolist()
        target = st.selectbox('Select target column', all_cols, index=len(all_cols)-1)
        features = st.multiselect('Select feature columns', [c for c in all_cols if c!=target], default=[c for c in all_cols if c!=target])
        algo = st.radio('Choose algorithm', ('Logistic Regression', 'Random Forest', 'Support Vector Machine'))
        if algo=='Random Forest':
            n_est = st.slider('n_estimators',10,200,50,step=10)
            max_d = st.slider('max_depth',1,20,5)
        test_size = st.slider('Test set size (%)',10,50,20)/100
        if st.button('Train Model'):
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.metrics import f1_score, classification_report, confusion_matrix
            import pickle
            X=data[features];y=data[target]
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=test_size,random_state=42)

            if algo == 'Logistic Regression':
                model = LogisticRegression(random_state=42)
            elif algo == 'Random Forest':
                model = RandomForestClassifier(n_estimators=n_est,max_depth=max_d,random_state=42)
            else:
                model = SVC(random_state=42)

            model.fit(X_tr,y_tr)
            p_tr, p_te = model.predict(X_tr), model.predict(X_te)
            st.subheader('Training Results')
            st.write(f"Test F1 Score: {f1_score(y_te, p_te, average='weighted'):.4f}")
            st.text('Classification Report:')
            st.text(classification_report(y_te, p_te))

            st.subheader('Confusion Matrix')
            import matplotlib.pyplot as plt
            import seaborn as sns
            cm = confusion_matrix(y_te, p_te, labels=model.classes_)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

            st.download_button('Download Model (.pkl)', pickle.dumps(model), 'model.pkl', 'application/octet-stream')
    else:
        st.info("Please load data in the 'Load Data' tab first.")
