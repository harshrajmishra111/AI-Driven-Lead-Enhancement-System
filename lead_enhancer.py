import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from textblob import TextBlob
from tabulate import tabulate
from colorama import init, Fore, Style
import re
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Initialize colorama
init()

# Load and validate dataset
def load_data(path):
    try:
        df = pd.read_csv('/Users/harshrajmishra/Documents/Caprae Capital/leads_data.csv')
        required_columns = ['email', 'job_title', 'company', 'bio', 'industry', 'revenue_estimate', 'contract_status', 'contract_value']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"{Fore.RED}Missing columns: {missing_cols}{Style.RESET_ALL}")
            return None
        df['revenue_estimate'] = df['revenue_estimate'].clip(lower=0)
        df['contract_value'] = df['contract_value'].clip(lower=0)
        valid_job_titles = ['CEO', 'Founder', 'CFO', 'VP of Sales', 'Marketing Director', 'Operations Manager', 'CTO', 'President']
        df = df[df['job_title'].isin(valid_job_titles)]
        if df.empty:
            print(f"{Fore.RED}No valid leads after validation.{Style.RESET_ALL}")
            return None
        return df
    except Exception as e:
        print(f"{Fore.RED}Error loading dataset: {e}{Style.RESET_ALL}")
        return None

# Advanced data cleaning
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
        print(f"{Fore.RED}No leads remain after cleaning.{Style.RESET_ALL}")
        return None
    return df

# Feature engineering
def engineer_features(df, contract_weight=0.5, status_weight=0.3, revenue_weight=0.2):
    df['revenue_to_contract_ratio'] = df.apply(
        lambda x: x['revenue_estimate'] / x['contract_value'] if x['contract_value'] > 0 else 0, axis=1
    )
    df['sentiment_score'] = df['bio'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['is_key_decision_maker'] = df['job_title'].apply(lambda x: 1 if x in ['CEO', 'Founder', 'President'] else 0)
    df['bio_length'] = df['bio'].str.len()
    df['contract_viability'] = df.apply(
        lambda x: contract_weight * (x['contract_value'] / 5000000) + 
                  status_weight * (1 if x['contract_status'] == 'Active' else 0.5 if x['contract_status'] == 'Pending' else 0) + 
                  revenue_weight * (x['revenue_estimate'] / 50000000), axis=1
    )
    return df

# Preprocess data
def preprocess_data(df):
    categorical_cols = ['job_title', 'industry', 'contract_status']
    numerical_cols = ['revenue_estimate', 'contract_value', 'revenue_to_contract_ratio', 'sentiment_score', 'is_key_decision_maker', 'contract_viability', 'bio_length']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ])
    X = preprocessor.fit_transform(df)
    return X, preprocessor

# Generate numerical labels with adjusted thresholds
def generate_labels(df):
    labels = []
    np.random.seed(42)
    for _, row in df.iterrows():
        base_score = 0
        if row['contract_viability'] > 0.45 and row['is_key_decision_maker'] == 1:
            base_score = 2
        elif row['contract_value'] > 650000 or row['sentiment_score'] > 0.1:
            base_score = 1
        else:
            base_score = 0
        noise = np.random.uniform(-0.2, 0.2)
        if base_score == 2 and noise < -0.15:
            base_score = 1
        elif base_score == 1 and noise > 0.15:
            base_score = 2
        labels.append(base_score)
    return np.array(labels)

# Train Logistic Regression model
def train_model(X, y):
    model = LogisticRegression(max_iter=1000, C=0.3, random_state=42)
    model.fit(X, y)
    return model

# Score leads
def score_leads(model, X, df, sentiment_weight=0.2):
    scores = model.predict_proba(X)
    high_idx = 2
    high_prob = scores[:, high_idx]
    df['ml_score'] = [round((1 - sentiment_weight) * prob + sentiment_weight * row['sentiment_score'], 2) for prob, row in zip(high_prob, df.to_dict('records'))]
    df['ml_score'] = df['ml_score'].clip(0, 1)
    df['confidence'] = [round(prob, 2) for prob in high_prob]
    return df, model

# Cluster leads
def cluster_leads(X, df, n_clusters=3):  # Fixed at 3 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    return df, kmeans

# Industry trends
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
def feature_importance(model, preprocessor):
    feature_names = preprocessor.get_feature_names_out()
    importance = np.abs(model.coef_[2])
    return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False)

# Generate summary report
def generate_report(df, trends, feat_imp, filename='lead_report.txt'):
    top_leads = df[df['ml_score'] > 0.7][['email', 'job_title', 'company', 'industry', 'ml_score', 'confidence', 'contract_status', 'contract_value', 'contract_rank', 'cluster']]
    report = (f"Top Leads Report (Score > 0.7)\n\n{tabulate(top_leads.head(10), headers='keys', tablefmt='grid')}\n\n"
              f"Total Leads: {len(top_leads)}\n\nIndustry Trends:\n{tabulate(trends, headers='keys', tablefmt='grid')}\n\n"
              f"Feature Importance:\n{tabulate(feat_imp.head(10), headers='keys', tablefmt='grid')}")
    with open(filename, 'w') as f:
        f.write(report)
    print(f"{Fore.GREEN}Report saved as {filename}{Style.RESET_ALL}")

# Evaluate models
def evaluate_models(X, y, model, df, kmeans):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{Fore.YELLOW}Logistic Regression Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f}){Style.RESET_ALL}")
    
    y_pred = model.predict(X)
    print(f"{Fore.YELLOW}Logistic Regression Classification Report (Full Data):{Style.RESET_ALL}")
    print(classification_report(y, y_pred, target_names=['Low', 'Medium', 'High']))
    print(f"{Fore.YELLOW}Logistic Regression Confusion Matrix (Full Data):{Style.RESET_ALL}")
    print(confusion_matrix(y, y_pred))
    
    silhouette_scores = {}
    for n in [3]:  # Only evaluate 3 clusters as per your preference
        kmeans_n = KMeans(n_clusters=n, random_state=42)
        cluster_labels = kmeans_n.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores[n] = score
    print(f"{Fore.YELLOW}KMeans Silhouette Scores:{Style.RESET_ALL}")
    for n, score in silhouette_scores.items():
        print(f"Clusters: {n}, Silhouette Score: {score:.3f}")

# Interactive CLI
def interactive_cli(df):
    contract_weight, status_weight, revenue_weight, sentiment_weight = 0.5, 0.3, 0.2, 0.2
    df = engineer_features(df, contract_weight, status_weight, revenue_weight)
    X, preprocessor = preprocess_data(df)
    y = generate_labels(df)
    df, model = score_leads(train_model(X, y), X, df, sentiment_weight)
    df, kmeans = cluster_leads(X, df, n_clusters=3)
    df['contract_rank'] = df['contract_value'].rank(ascending=False, method='dense')
    trends = industry_trends(df)
    feat_imp = feature_importance(model, preprocessor)

    while True:
        print(f"\n{Fore.CYAN}Options: filter, view, export, report, trends, importance, evaluate, exit{Style.RESET_ALL}")
        choice = input("Select option: ").lower()
        if choice == 'exit':
            break
        elif choice == 'filter':
            industry = input(f"{Fore.CYAN}Enter industry (or 'All'): {Style.RESET_ALL}").capitalize()
            contract_status = input(f"{Fore.CYAN}Enter contract status (or 'All'): {Style.RESET_ALL}").capitalize()
            cluster = input(f"{Fore.CYAN}Enter cluster (or 'All'): {Style.RESET_ALL}")
            try:
                min_score = float(input(f"{Fore.CYAN}Enter minimum ML score (0.0 to 1.0) [default 0.0]: {Style.RESET_ALL}") or 0.0)
            except ValueError:
                min_score = 0.0
            filtered_df = df
            if industry != 'All' and industry in df['industry'].values:
                filtered_df = filtered_df[filtered_df['industry'] == industry]
            if contract_status != 'All' and contract_status in df['contract_status'].values:
                filtered_df = filtered_df[filtered_df['contract_status'] == contract_status]
            if cluster != 'All' and cluster in df['cluster'].astype(str).values:
                filtered_df = filtered_df[filtered_df['cluster'] == int(cluster)]
            filtered_df = filtered_df[filtered_df['ml_score'] >= min_score]
            print(f"{Fore.YELLOW}{tabulate(filtered_df.head(10), headers='keys', tablefmt='grid')}{Style.RESET_ALL}")
        elif choice == 'view':
            print(f"{Fore.YELLOW}{tabulate(df.head(10), headers='keys', tablefmt='grid')}{Style.RESET_ALL}")
        elif choice == 'export':
            industry = input(f"{Fore.CYAN}Enter industry to export (or 'All'): {Style.RESET_ALL}").capitalize()
            filtered_df = df
            if industry != 'All' and industry in df['industry'].values:
                filtered_df = filtered_df[filtered_df['industry'] == industry]
            filtered_df.to_csv('enhanced_leads.csv', index=False)
            print(f"{Fore.GREEN}Exported to enhanced_leads.csv{Style.RESET_ALL}")
        elif choice == 'report':
            generate_report(df, trends, feat_imp)
        elif choice == 'trends':
            print(f"{Fore.YELLOW}Industry Trends:{Style.RESET_ALL}")
            trends.to_csv('industry_trends.csv')
            print(f"{Fore.GREEN}Trends saved as industry_trends.csv{Style.RESET_ALL}")
            print(tabulate(trends, headers='keys', tablefmt='grid'))
        elif choice == 'importance':
            print(f"{Fore.YELLOW}Feature Importance:{Style.RESET_ALL}")
            print(tabulate(feat_imp.head(10), headers='keys', tablefmt='grid'))
        elif choice == 'evaluate':
            evaluate_models(X, y, model, df, kmeans)
        else:
            print(f"{Fore.RED}Invalid option. Try again.{Style.RESET_ALL}")

# Main function
def main():
    print(f"{Fore.BLUE}Processing leads...{Style.RESET_ALL}")
    df = load_data('/Users/harshrajmishra/Documents/Caprae Capital/leads_data.csv')
    if df is None:
        return
    df = clean_data(df)
    if df is None:
        return
    interactive_cli(df)

if __name__ == '__main__':
    main()