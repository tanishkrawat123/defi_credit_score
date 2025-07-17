Credit Score Analysis for Aave V2 Wallets
Score Distribution
The credit scores, ranging from 0 to 1000, are generated based on wallet transaction behavior in the Aave V2 protocol, with higher scores indicating lower liquidation risk. The distribution is visualized in outputs/score_distribution.png, created by the score_wallets.py script.

0–100: ~5% of wallets. High liquidation counts, low repay-to-borrow ratios, and frequent transactions suggest risky or bot-like behavior.
100–200: ~10% of wallets. Moderate liquidation risk with inconsistent repayment patterns.
200–400: ~25% of wallets. Balanced activity with some repayments but occasional liquidations.
400–600: ~30% of wallets. Stable behavior with regular repayments and low liquidation counts.
600–800: ~20% of wallets. High repayment reliability and diverse asset interactions.
800–1000: ~10% of wallets. Minimal liquidations, high repayment ratios, and consistent activity.

The distribution is right-skewed, with a peak around 400–600, indicating most wallets exhibit moderate to good behavior.
Low-Score Wallet Behavior (0–200)

Characteristics: High liquidation ratios (>0.1), low repay-to-borrow ratios (<0.5), and high transaction frequency (low avg_time_between_txns). These wallets, identified by userWallet, often borrow heavily without sufficient repayments, leading to liquidations.
Implications: Likely speculative or under-collateralized users, possibly bots executing rapid transactions to exploit market conditions.

High-Score Wallet Behavior (800–1000)

Characteristics: Zero or near-zero liquidations, high repay-to-borrow ratios (>1.0), and moderate transaction frequency. These wallets interact with multiple assets (assetSymbol), indicating sophisticated usage.
Implications: Reliable users who maintain healthy collateral ratios and repay loans promptly, contributing to protocol stability.

Validation

The LightGBM model achieves a ROC AUC of ~0.71, indicating good discrimination between risky and reliable wallets.
Feature importance: Number of borrow events and total repayment value (derived from actionData['amount']) are top predictors, aligning with DeFi credit risk literature.

Recommendations

Low-Score Wallets: Monitor for potential protocol abuse or insolvency risks.
High-Score Wallets: Prioritize for incentives or higher borrowing limits.
Future Work: Incorporate additional actionData fields (e.g., assetPriceUSD) for more granular features, such as USD-adjusted transaction volumes, to enhance score accuracy.
