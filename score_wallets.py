import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

# Feature engineering
def engineer_features(df):
    # Extract amount and type from actionData
    if 'actionData' in df.columns:
        def parse_amount(x):
            try:
                if isinstance(x, dict):
                    amount = x.get('amount', 0)
                    return float(amount) if amount else 0.0
                return 0.0
            except (ValueError, TypeError):
                return 0.0
        df['amount'] = df['actionData'].apply(parse_amount)
        
        # Extract transaction type and normalize to lowercase
        df['action_type'] = df['actionData'].apply(lambda x: x.get('type', '').lower() if isinstance(x, dict) else '')
    
    # Convert timestamps to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Aggregate by wallet
    features = df.groupby('userWallet').agg({
        'amount': ['count', 'sum'],
        'action_type': [
            lambda x: (x == 'deposit').sum(),
            lambda x: (x == 'borrow').sum(),
            lambda x: (x == 'repay').sum(),
            lambda x: (x == 'liquidationcall').sum()
        ]
    }).reset_index()
    
    # Flatten column names
    features.columns = [
        'userWallet', 'txn_count', 'total_volume',
        'deposit_count', 'borrow_count', 'repay_count', 'liquidation_count'
    ]
    
    # Additional features
    features['repay_to_borrow_ratio'] = features['repay_count'] / (features['borrow_count'] + 1e-6)
    features['liquidation_ratio'] = features['liquidation_count'] / (features['txn_count'] + 1e-6)
    features['avg_txn_volume'] = features['total_volume'] / (features['txn_count'] + 1e-6)
    
    # Bot-like behavior: high transaction frequency in short time
    if 'timestamp' in df.columns:
        txn_freq = df.groupby('userWallet').apply(
            lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds() / (x.shape[0] + 1e-6)
        ).reset_index(name='avg_time_between_txns')
        features = features.merge(txn_freq, on='userWallet')
    
    # Pool diversity (using assetSymbol instead of reserve)
    if 'actionData' in df.columns:
        df['reserve'] = df['actionData'].apply(lambda x: x.get('assetSymbol', None) if isinstance(x, dict) else None)
        features['pool_diversity'] = df.groupby('userWallet')['reserve'].nunique().reset_index(name='pool_diversity')['pool_diversity']
    
    # Fill missing values
    features = features.fillna(0)
    return features

# Train model and assign scores
def train_and_score(features):
    # Target: 1 if no liquidation, 0 if liquidation occurred
    features['target'] = (features['liquidation_count'] == 0).astype(int)
    
    # Features for modeling
    X = features.drop(['userWallet', 'target'], axis=1)
    y = features['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    
    # Train LightGBM model
    model = LGBMClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities
    scores = model.predict_proba(X_scaled)[:, 1] * 1000
    features['credit_score'] = np.clip(scores, 0, 1000)
    
    return features[['userWallet', 'credit_score']], model, X.columns

# Plot score distribution
def plot_score_distribution(scores_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(scores_df['credit_score'], bins=10, kde=True)
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.savefig('outputs/score_distribution.png')
    plt.close()

# Main execution
def main():
    os.makedirs('outputs', exist_ok=True)
    df = load_data('data/user-wallet-transactions.json')
    features = engineer_features(df)
    scores_df, model, feature_names = train_and_score(features)
    scores_df.to_csv('outputs/wallet_scores.csv', index=False)
    plot_score_distribution(scores_df)
    print("Scoring complete. Results saved to outputs/wallet_scores.csv")

if __name__ == '__main__':
    main()