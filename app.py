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

    # Load Raw Data
    raw_files = glob.glob(os.path.join(data_dir, 'metal', '*', '*', '*', '*', '*', 'summary_results.csv'))
    if raw_files:
        df_list = []
        for file in raw_files:
            try:
                parts = file.split(os.sep)
                metal_type = parts[-6]
                sample = parts[-4]
                direction = parts[-3]
                timestamp_str = parts[-2]

                temp_df = pd.read_csv(file)
                temp_df['metal_type'] = metal_type
                temp_df['sample'] = sample
                temp_df['direction'] = direction
                temp_df['timestamp'] = pd.to_datetime(timestamp_str, format='%Y%m%d-%H%M%S')
                df_list.append(temp_df)
            except Exception as e:
                st.warning(f"Could not process raw data file {file}: {e}")
        if df_list:
            st.session_state.raw_df = pd.concat(df_list, ignore_index=True)
            st.success("Raw data loaded successfully!")

    # Load Calibrated Data
    calibrated_files = glob.glob(os.path.join(data_dir, 'metal', '*', '*', '*', '*', '*_CALIBRATED.csv'))
    if calibrated_files:
        df_list = []
        for file in calibrated_files:
            try:
                parts = file.split(os.sep)
                filename = parts[-1]
                metal_type = filename.split('_')[0]
                sample = filename.split('_')[1]
                direction = filename.split('_')[2]

                temp_df = pd.read_csv(file)
                temp_df['metal_type'] = metal_type
                temp_df['sample'] = sample
                temp_df['direction'] = direction
                df_list.append(temp_df)
            except Exception as e:
                st.warning(f"Could not process calibrated data file {file}: {e}")
        if df_list:
            st.session_state.calibrated_df = pd.concat(df_list, ignore_index=True)
            st.success("Calibrated data loaded successfully!")

    # Load Eigenvalues Data
    eigenvalues_files = glob.glob(os.path.join(data_dir, 'metal', '*', '*', '*', '*_Eigenvalues.csv'))
    if eigenvalues_files:
        df_list = []
        for file in eigenvalues_files:
            try:
                parts = file.split(os.sep)
                filename = parts[-1]
                metal_type = filename.split('_')[0]
                sample = filename.split('_')[2]

                temp_df = pd.read_csv(file)
                temp_df['metal_type'] = metal_type
                temp_df['sample'] = sample
                df_list.append(temp_df)
            except Exception as e:
                st.warning(f"Could not process eigenvalues data file {file}: {e}")
        if df_list:
            st.session_state.eigenvalues_df = pd.concat(df_list, ignore_index=True)
            st.success("Eigenvalues data loaded successfully!")

# --- Dataset Selector ---
st.sidebar.header("Dataset Selection")
dataset_options = []
if 'raw_df' in st.session_state:
    dataset_options.append("Raw Data")
if 'calibrated_df' in st.session_state:
    dataset_options.append("Calibrated Data")
if 'eigenvalues_df' in st.session_state:
    dataset_options.append("Eigenvalues Data")

if dataset_options:
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)

    if selected_dataset == "Raw Data":
        st.session_state.master_df = st.session_state.raw_df
    elif selected_dataset == "Calibrated Data":
        st.session_state.master_df = st.session_state.calibrated_df
    elif selected_dataset == "Eigenvalues Data":
        st.session_state.master_df = st.session_state.eigenvalues_df

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

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

    # Apply filters
    filtered_df = df[
        (df['metal_type'].isin(selected_metals)) &
        (df['sample'].isin(selected_samples)) &
        (df['direction'].isin(selected_directions)) &
        (df['timestamp'].dt.date >= start_date) &
        (df['timestamp'].dt.date <= end_date)
    ]
    st.session_state.filtered_df = filtered_df

# --- Tabs ---
tab_load, tab_train = st.tabs(["📤 Load Data", "🤖 Train Model"])

with tab_load:
    st.header("Interactive Plot")
    if 'filtered_df' in st.session_state and not st.session_state.filtered_df.empty:
        df = st.session_state.filtered_df

        st.subheader("Data Preview")
        st.dataframe(df)

        st.subheader("Create Plot")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
        default_y = [c for c in numeric_cols if c != x_axis]
        y_axes = st.multiselect("Select Y-axis", numeric_cols, default=default_y[:2])

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
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect('Select feature columns', [c for c in numeric_cols if c!=target], default=[c for c in numeric_cols if c!=target])
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
