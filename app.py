import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- App Config ---
st.set_page_config(page_title="ML GUI: Data Cleaning & Model Training",
                   layout="wide")
st.title("📊 ML GUI: Data Cleaning & Model Training")

# --- Tabs ---
tab_clean, tab_train = st.tabs(["🧹 Clean Data", "🤖 Train Model"])

with tab_clean:
    st.header("Data Cleaning & Anomaly Detection")
    uploaded = st.file_uploader("Upload raw CSV", type=["csv"])
    if uploaded:
        df_full = pd.read_csv(uploaded)
        freq_col = 'Frequency'
        # Choose real and imag series
        numeric_cols = [c for c in df_full.select_dtypes(include=np.number).columns if c != freq_col]
        default_real = [c for c in numeric_cols if 'Real' in c]
        default_imag = [c for c in numeric_cols if 'Imag' in c]
        defaults = []
        if default_real: defaults.append(default_real[0])
        if default_imag: defaults.append(default_imag[0])
        selected = st.multiselect(
            'Select two series to analyze (Real & Imag)',
            numeric_cols,
            default=defaults if defaults else numeric_cols[:2]
        )
        if len(selected) != 2:
            st.warning('Please select exactly two series.')
            st.stop()
        # Compute sliding-3 predictions for both series
        freqs_full = df_full[freq_col].values
        logf_full = np.log(freqs_full)
        for series in selected:
            y_full = df_full[series].values
            preds_full = np.full_like(y_full, np.nan, dtype=float)
            for i in range(1, len(df_full)-1):
                x1, x2, x3 = logf_full[i-1], logf_full[i], logf_full[i+1]
                y1, y3 = y_full[i-1], y_full[i+1]
                m = (y3 - y1) / (x3 - x1)
                preds_full[i] = y1 + m * (x2 - x1)
            # first and last points
            if len(df_full) >= 3:
                # first
                x1, x3 = logf_full[1], logf_full[2]
                y1, y3 = y_full[1], y_full[2]
                m0 = (y3 - y1)/(x3-x1)
                preds_full[0] = y1 + m0*(logf_full[0]-x1)
                # last
                x_prev2, x_prev1 = logf_full[-3], logf_full[-2]
                y_prev2, y_prev1 = y_full[-3], y_full[-2]
                m_last = (y_prev1-y_prev2)/(x_prev1-x_prev2)
                preds_full[-1] = y_prev1 + m_last*(logf_full[-1]-x_prev1)
            df_full[f'{series}_pred'] = preds_full
                        # Frequency range filtering and error thresholding
        st.subheader('Filter & Error Threshold')
        # Calculate errors for both series and max error
        df_full['error_0'] = (df_full[selected[0]] - df_full[f'{selected[0]}_pred']).abs()
        df_full['error_1'] = (df_full[selected[1]] - df_full[f'{selected[1]}_pred']).abs()
        df_full['error_max'] = df_full[['error_0','error_1']].max(axis=1)

        # Define frequency bounds and defaults
        min_bound = float(freqs_full.min())
        max_bound = float(freqs_full.max())
        default_min = 500.0 if 500.0 >= min_bound else min_bound
        default_max = 50000.0 if 50000.0 <= max_bound else max_bound

        # Inputs: manual freq range and error threshold
        fcol1, fcol2, fcol3 = st.columns([1,1,1])
        with fcol1:
            st.markdown('**Frequency Range (Hz)**')
            freq_min = st.number_input('Min Frequency', value=default_min, min_value=min_bound, max_value=max_bound)
            freq_max = st.number_input('Max Frequency', value=default_max, min_value=min_bound, max_value=max_bound)
        with fcol2:
            st.markdown('**Error Threshold**')
            err_thr = st.number_input('Max Error Threshold', min_value=0.0, value=float(df_full['error_max'].max()), step=0.01)
        with fcol3:
            st.markdown('**Top 5 Frequencies by Error**')
            top5 = df_full.nlargest(5, 'error_max')[[freq_col,'error_0','error_1','error_max']]
            st.dataframe(top5)

        # apply filters
        mask = (df_full[freq_col]>=freq_min) & (df_full[freq_col]<=freq_max) & (df_full['error_max']<=err_thr)
        df = df_full[mask].reset_index(drop=True)
        freqs = df[freq_col].values
        mask = (df_full[freq_col]>=freq_min)&(df_full[freq_col]<=freq_max)&(df_full['error_max']<=err_thr)
        df = df_full[mask].reset_index(drop=True)
        freqs = df[freq_col].values
        # Context selection        

        st.subheader('Inspect Specific Row')
        idx = st.slider('Row index (filtered data)', 0, len(df)-1, 0)
        ctx = 10
        start, end = max(0,idx-ctx), min(len(df)-1,idx+ctx)
        col1, col2, col3 = st.columns([1,2,1])
        # Raw data
        with col1:
            st.subheader('Raw Data (Context)')
            raw_ctx = df[[freq_col]+selected].loc[start:end]
            styled_raw = raw_ctx.style.apply(
                lambda r: ['font-weight:bold;color:lightblue' if r.name==idx else '' for _ in r], axis=1
            )
            st.dataframe(styled_raw, use_container_width=True)
        # Graph
        with col2:
            st.subheader('Actual vs Predicted (Interactive)')
            fig = go.Figure()
            for series in selected:
                fig.add_trace(go.Scatter(
                    x=freqs, y=df[series].values,
                    mode='lines+markers', name=f'{series} Actual',
                    line=dict(dash='solid'), marker=dict(size=6)
                ))
                fig.add_trace(go.Scatter(
                    x=freqs, y=df[f'{series}_pred'].values,
                    mode='lines+markers', name=f'{series} Pred',
                    line=dict(dash='dash'), marker=dict(size=6)
                ))
                # highlight
                fig.add_trace(go.Scatter(
                    x=[freqs[idx]], y=[df[series].iloc[idx]],
                    mode='markers', name=f'Selected {series}',
                    marker=dict(size=12, symbol='circle')
                ))
            fig.update_xaxes(type='log', title_text='Frequency (Hz)')
            fig.update_layout(margin=dict(l=40,r=40,t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)
        # Predicted table
        with col3:
            st.subheader('Predicted Data (Context)')
            pred_cols = [f'{s}_pred' for s in selected]
            pred_ctx = df[[freq_col]+pred_cols].loc[start:end]
            styled_pred = pred_ctx.style.apply(
                lambda r: ['font-weight:bold;color:lightblue' if r.name==idx else '' for _ in r], axis=1
            )
            st.dataframe(styled_pred, use_container_width=True)
        # Download full cleaned
        csv = df_full.to_csv(index=False).encode()
        st.download_button('📥 Download Cleaned CSV (full)', csv, 'cleaned.csv', 'text/csv')

with tab_train:
    # unchanged
    st.header('Model Training')
    st.info('Use cleaned CSV (from Clean Data tab) or upload your own processed CSV.')
    train_file = st.file_uploader('Upload cleaned CSV', type=['csv'], key='train')
    if train_file:
        data = pd.read_csv(train_file)
        st.subheader('Dataset Preview')
        st.dataframe(data.head(), use_container_width=True)
        all_cols = data.columns.tolist()
        target = st.selectbox('Select target column', all_cols)
        features = st.multiselect('Select feature columns', [c for c in all_cols if c!=target], default=[c for c in all_cols if c!=target][:3])
        algo = st.radio('Choose algorithm', ('Linear Regression','Random Forest'))
        if algo=='Random Forest':
            n_est = st.slider('n_estimators',10,200,50,step=10)
            max_d = st.slider('max_depth',1,20,5)
        test_size = st.slider('Test set size (%)',10,50,20)/100
        if st.button('Train Model'):
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error,r2_score
            import pickle
            X=data[features];y=data[target]
            X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=test_size,random_state=42)
            model = LinearRegression() if algo=='Linear Regression' else RandomForestRegressor(n_estimators=n_est,max_depth=max_d,random_state=42)
            model.fit(X_tr,y_tr)
            p_tr, p_te = model.predict(X_tr), model.predict(X_te)
            st.subheader('Training Results')
            st.write(f'Train MSE: {mean_squared_error(y_tr,p_tr):.4f}, R²: {r2_score(y_tr,p_tr):.4f}')
            st.write(f'Test MSE: {mean_squared_error(y_te,p_te):.4f}, R²: {r2_score(y_te,p_te):.4f}')
            import matplotlib.pyplot as plt
            fig2,ax=plt.subplots(1,2,figsize=(12,4))
            ax[0].scatter(y_tr,p_tr,alpha=0.7);ax[0].plot([y_tr.min(),y_tr.max()],[y_tr.min(),y_tr.max()],'k--');ax[0].set_title('Train')
            ax[1].scatter(y_te,p_te,alpha=0.7);ax[1].plot([y_te.min(),y_te.max()],[y_te.min(),y_te.max()],'k--');ax[1].set_title('Test')
            st.pyplot(fig2)
            st.download_button('Download Model (.pkl)', pickle.dumps(model), 'model.pkl', 'application/octet-stream')
