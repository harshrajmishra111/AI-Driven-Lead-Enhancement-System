import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from textblob import TextBlob
import plotly.express as px
import plotly.io as pio
import re

# Streamlit app configuration
st.set_page_config(page_title="SaaSquatch Lead Enhancer", layout="wide")

# Load and validate dataset
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv('/Users/harshrajmishra/Documents/Caprae Capital/leads_data.csv')
        required_columns = ['email', 'job_title', 'company', 'bio', 'industry', 'revenue_estimate', 'contract_status', 'contract_value']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            return None
        df['revenue_estimate'] = df['revenue_estimate'].clip(lower=0)
        df['contract_value'] = df['contract_value'].clip(lower=0)
        valid_job_titles = ['CEO', 'Founder', 'CFO', 'VP of Sales', 'Marketing Director', 'Operations Manager', 'CTO', 'President']
        df = df[df['job_title'].isin(valid_job_titles)]
        if df.empty:
            st.error("No valid leads after validation.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Advanced data cleaning
@st.cache_data
def clean_data(df):
    df = df.drop_duplicates(subset=['email'], keep='first')
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    df['is_valid_email'] = df['email'].apply(lambda x: bool(email_pattern.match(str(x))))
    df = df[df['is_valid_email']].drop(columns=['is_valid_email'])
    df['contract_status'] = df['contract_status'].fillna('None').astype(str)
    df['contract_value'] = df['contract_value'].fillna(0).astype(float)
    df['bio'] = df['bio'].fillna('').astype(str)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    numerical_cols = ['revenue_estimate', 'contract_value']
    df['is_outlier'] = iso_forest.fit_predict(df[numerical_cols]) == -1
    df = df[~df['is_outlier']].drop(columns=['is_outlier'])
    if df.empty:
        st.error("No leads remain after cleaning.")
        return None
    return df

# Feature engineering
@st.cache_data
def engineer_features(df, contract_weight=0.5, status_weight=0.3, revenue_weight=0.2):
    df['revenue_to_contract_ratio'] = df.apply(
        lambda x: x['revenue_estimate'] / x['contract_value'] if x['contract_value'] > 0 else 0, axis=1
    )
    df['sentiment_score'] = df['bio'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['is_key_decision_maker'] = df['job_title'].apply(lambda x: 1 if x in ['CEO', 'Founder', 'President'] else 0)
    df['source'] = [np.random.choice(['LinkedIn', 'Crunchbase', 'Google Maps']) for _ in range(len(df))]
    df['contract_viability'] = df.apply(
        lambda x: contract_weight * (x['contract_value'] / 5000000) + 
                  status_weight * (1 if x['contract_status'] == 'Active' else 0.5 if x['contract_status'] == 'Pending' else 0) + 
                  revenue_weight * (x['revenue_estimate'] / 50000000), axis=1
    )
    return df

# Preprocess data
@st.cache_data
def preprocess_data(df):
    categorical_cols = ['job_title', 'industry', 'contract_status', 'source']
    numerical_cols = ['revenue_estimate', 'contract_value', 'revenue_to_contract_ratio', 'sentiment_score', 'is_key_decision_maker', 'contract_viability']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ])
    X = preprocessor.fit_transform(df)
    return X, preprocessor

# Generate numerical labels
def generate_labels(df):
    labels = []
    for _, row in df.iterrows():
        if row['contract_viability'] > 0.7 and row['is_key_decision_maker'] == 1:
            labels.append(2)  # High
        elif row['contract_value'] > 1000000 or row['sentiment_score'] > 0.3:
            labels.append(1)  # Medium
        else:
            labels.append(0)  # Low
    return np.array(labels)

# Train XGBoost model
@st.cache_resource
def train_model(X, y):
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    model.fit(X, y)
    return model

# Score leads
@st.cache_data
def score_leads(_model, X, df, sentiment_weight=0.2):
    scores = _model.predict_proba(X)
    high_idx = 2  # Index for 'High' class (2)
    high_prob = scores[:, high_idx]
    df['ml_score'] = [round((1 - sentiment_weight) * prob + sentiment_weight * row['sentiment_score'], 2) for prob, row in zip(high_prob, df.to_dict('records'))]
    df['ml_score'] = df['ml_score'].clip(0, 1)
    df['confidence'] = [round(prob, 2) for prob in high_prob]
    return df

# Cluster leads
@st.cache_data
def cluster_leads(X, df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    return df, kmeans

# Industry trends
@st.cache_data
def industry_trends(df):
    trends = df.groupby('industry').agg({
        'ml_score': 'mean',
        'contract_value': 'mean',
        'revenue_estimate': 'mean',
        'email': 'count',
        'cluster': lambda x: x.mode()[0] if not x.empty else -1
    }).round(2)
    trends.columns = ['Avg Score', 'Avg Contract Value', 'Avg Revenue', 'Lead Count', 'Dominant Cluster']
    return trends

# Feature importance
@st.cache_data
def feature_importance(_model, _preprocessor):
    feature_names = _preprocessor.get_feature_names_out()
    importance = _model.feature_importances_
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False)

# Streamlit app
st.title("SaaSquatch Lead Enhancer")
st.markdown("Interactive dashboard for prioritizing high-value acquisition leads.")

# Load and process data
df = load_data('/Users/harshrajmishra/Documents/Caprae Capital/leads_data.csv')
if df is not None:
    # Custom scoring weights
    st.sidebar.header("Scoring Weights")
    contract_weight = st.sidebar.slider("Contract Value Weight", 0.0, 1.0, 0.5)
    status_weight = st.sidebar.slider("Contract Status Weight", 0.0, 1.0, 0.3)
    revenue_weight = st.sidebar.slider("Revenue Weight", 0.0, 1.0, 0.2)
    sentiment_weight = st.sidebar.slider("Sentiment Weight", 0.0, 0.5, 0.2)
    
    # Normalize weights
    total_weight = contract_weight + status_weight + revenue_weight
    if total_weight > 0:
        contract_weight /= total_weight
        status_weight /= total_weight
        revenue_weight /= total_weight
    else:
        st.warning("Weights sum to zero, using defaults.")
        contract_weight, status_weight, revenue_weight = 0.5, 0.3, 0.2

    df = clean_data(df)
    if df is None:
        st.stop()
    df = engineer_features(df, contract_weight, status_weight, revenue_weight)
    X, preprocessor = preprocess_data(df)
    y = generate_labels(df)
    model = train_model(X, y)
    df = score_leads(model, X, df, sentiment_weight)
    df, kmeans = cluster_leads(X, df)
    df['contract_rank'] = df['contract_value'].rank(ascending=False, method='dense')
    trends = industry_trends(df)
    feat_imp = feature_importance(model, preprocessor)

    # Sidebar for filters
    st.sidebar.header("Filter Leads")
    industry = st.sidebar.selectbox("Industry", ['All'] + list(df['industry'].unique()))
    contract_status = st.sidebar.selectbox("Contract Status", ['All'] + list(df['contract_status'].unique()))
    source = st.sidebar.selectbox("Source", ['All'] + list(df['source'].unique()))
    cluster = st.sidebar.selectbox("Cluster", ['All'] + list(df['cluster'].unique()))
    min_score = st.sidebar.slider("Minimum ML Score", 0.0, 1.0, 0.0)

    # Filter data
    filtered_df = df
    if industry != 'All':
        filtered_df = filtered_df[filtered_df['industry'] == industry]
    if contract_status != 'All':
        filtered_df = filtered_df[filtered_df['contract_status'] == contract_status]
    if source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == source]
    if cluster != 'All':
        filtered_df = filtered_df[filtered_df['cluster'] == cluster]
    filtered_df = filtered_df[filtered_df['ml_score'] >= min_score]

    # Display leads
    st.subheader("Top Leads")
    st.dataframe(filtered_df[['email', 'job_title', 'company', 'industry', 'source', 'ml_score', 'confidence', 'contract_status', 'contract_value', 'contract_rank', 'cluster']].head(10))

    # Visualization
    st.subheader("Lead Distribution")
    fig = px.scatter(
        filtered_df,
        x='revenue_estimate',
        y='contract_value',
        color='ml_score',
        size='confidence',
        text='email',
        hover_data=['job_title', 'company', 'industry', 'source', 'cluster'],
        title="Lead Score vs. Revenue and Contract Value",
        color_continuous_scale='Viridis'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
    
    # Download plot
    if st.button("Download Plot as PNG"):
        pio.write_image(fig, file='lead_distribution.png', format='png')
        with open('lead_distribution.png', 'rb') as f:
            st.download_button("Download PNG", f, file_name='lead_distribution.png')

    # Industry trends
    st.subheader("Industry Trends")
    st.dataframe(trends)
    if st.button("Export Trends"):
        trends.to_csv('industry_trends.csv')
        st.success("Exported to industry_trends.csv")
        with open('industry_trends.csv', 'rb') as f:
            st.download_button("Download Trends CSV", f, file_name='industry_trends.csv')

    # Feature importance
    st.subheader("Feature Importance")
    st.dataframe(feat_imp.head(10))

    # Export and report
    st.subheader("Export & Report")
    if st.button("Export Filtered Leads"):
        filtered_df.to_csv('enhanced_leads.csv', index=False)
        st.success("Exported to enhanced_leads.csv")
        with open('enhanced_leads.csv', 'rb') as f:
            st.download_button("Download CSV", f, file_name='enhanced_leads.csv')
    if st.button("Generate Report"):
        top_leads = filtered_df[filtered_df['ml_score'] > 0.7]
        report = f"Top Leads Report (Score > 0.7)\n\n{top_leads[['email', 'job_title', 'company', 'industry', 'source', 'ml_score', 'confidence', 'contract_status', 'contract_value', 'contract_rank', 'cluster']].head(10).to_markdown()}\n\nTotal Leads: {len(top_leads)}\n\nIndustry Trends:\n{trends.to_markdown()}\n\nFeature Importance:\n{feat_imp.head(10).to_markdown()}"
        with open('lead_report.txt', 'w') as f:
            f.write(report)
        st.success("Report saved as lead_report.txt")
        with open('lead_report.txt', 'rb') as f:
            st.download_button("Download Report", f, file_name='lead_report.txt')