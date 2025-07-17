DeFi Credit Scoring for Aave V2 Protocol
Overview
This project develops a machine learning model to assign credit scores (0–1000) to wallets interacting with the Aave V2 protocol, based on transaction-level data. Higher scores indicate reliable behavior (e.g., consistent repayments, low liquidation risk), while lower scores reflect risky or bot-like behavior. The model leverages on-chain data to predict the probability of non-liquidation, scaled to a credit score.
Methodology

Data Source: The input is a JSON file (user-wallet-transactions.json) containing transaction records with fields like userWallet, action, actionData (containing type, amount, assetSymbol), timestamp, and others.
Feature Engineering:
Transaction Frequency: Total transactions, deposits, borrows, repays, and liquidations per wallet, derived from actionData['type'].
Transaction Volume: Sum and average of transaction amounts, extracted from actionData['amount'] and converted to numeric values.
Repayment Behavior: Ratio of repays to borrows, based on actionData['type'] counts.
Liquidation Risk: Ratio of liquidations to total transactions.
Pool Diversity: Number of unique assets (actionData['assetSymbol']) interacted with.
Bot-like Behavior: Average time between transactions, calculated using timestamp.


Model: A LightGBM classifier predicts the probability of non-liquidation (target: 1 if no liquidations, 0 otherwise). Probabilities are scaled to 0–1000 for credit scores.
Validation: Out-of-time train-test split (33% test set) ensures robustness to unseen data. Feature importance is derived from model performance.

Architecture

Data Loading: Load JSON into a pandas DataFrame using load_data().
Feature Engineering: Extract amount and type from actionData, aggregate by userWallet to compute features.
Preprocessing: Standardize features and handle missing values or non-numeric data.
Modeling: Train LightGBM on engineered features, predicting non-liquidation probability.
Scoring: Scale probabilities to 0–1000 and save to wallet_scores.csv.
Analysis: Generate score distribution plot and behavioral analysis in analysis.md.

Processing Flow

Place user-wallet-transactions.json in the data/ folder.
Run score_wallets.py to process transactions, train the model, and generate outputs.
Outputs are saved to outputs/:
wallet_scores.csv: Contains userWallet and credit_score columns.
score_distribution.png: Visualizes the score distribution.


Review analysis.md for score distribution and wallet behavior insights.

Setup

Clone the repository: git clone <repo_url>
Install dependencies: pip install -r requirements.txt
Place user-wallet-transactions.json in data/.
Run: python score_wallets.py

Extensibility

Add new features (e.g., USD-adjusted volumes using actionData['assetPriceUSD']) by extending engineer_features().
Incorporate real-time data via APIs (e.g., The Graph) for dynamic scoring.
Experiment with other models (e.g., XGBoost) in train_and_score().

Dependencies
See requirements.txt for required Python packages (pandas, numpy, lightgbm, scikit-learn, matplotlib, seaborn).
Data Assumptions

The JSON file contains userWallet for wallet IDs, timestamp for transaction times, and actionData with type (e.g., 'Deposit', 'Borrow'), amount (as strings), and assetSymbol for assets.
Non-numeric amount values are converted to floats; missing values are set to 0.
