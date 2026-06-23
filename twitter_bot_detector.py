import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

#Simulate Twitter account features
np.random.seed(42)

def generate_accounts(n, is_bot=False):
    if is_bot:
        return {
            'follower_count': np.random.randint(0, 100, n),
            'following_count': np.random.randint(1000, 5000, n),
            'tweet_count': np.random.randint(5000, 50000, n),
            'account_age_days': np.random.randint(1, 180, n),
            'tweets_per_day': np.random.uniform(50, 200, n),
            'has_profile_pic': np.random.choice([0, 1], n, p=[0.6, 0.4]),
            'has_bio': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'verified': np.zeros(n),
            'avg_retweet_ratio': np.random.uniform(0.8, 1.0, n),
            'url_in_bio': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'default_profile_theme': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'reply_ratio': np.random.uniform(0.0, 0.05, n),
            'is_bot': np.ones(n, dtype=int)
        }
    else:
        return {
            'follower_count': np.random.randint(100, 10000, n),
            'following_count': np.random.randint(50, 2000, n),
            'tweet_count': np.random.randint(100, 5000, n),
            'account_age_days': np.random.randint(180, 3650, n),
            'tweets_per_day': np.random.uniform(0.5, 15, n),
            'has_profile_pic': np.random.choice([0, 1], n, p=[0.1, 0.9]),
            'has_bio': np.random.choice([0, 1], n, p=[0.2, 0.8]),
            'verified': np.random.choice([0, 1], n, p=[0.95, 0.05]),
            'avg_retweet_ratio': np.random.uniform(0.1, 0.5, n),
            'url_in_bio': np.random.choice([0, 1], n, p=[0.5, 0.5]),
            'default_profile_theme': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'reply_ratio': np.random.uniform(0.1, 0.4, n),
            'is_bot': np.zeros(n, dtype=int)
        }

bots = pd.DataFrame(generate_accounts(1000, is_bot=True))
humans = pd.DataFrame(generate_accounts(1000, is_bot=False))
df = pd.concat([bots, humans]).sample(frac=1, random_state=42).reset_index(drop=True)

#Feature engineering
df['follower_following_ratio'] = df['follower_count'] / (df['following_count'] + 1)
df['activity_score'] = df['tweets_per_day'] * (1 / (df['account_age_days'] + 1))

print(f" Dataset: {len(df)} accounts ({df['is_bot'].sum()} bots)")

X = df.drop('is_bot', axis=1)
y = df['is_bot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

#Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)
preds = rf.predict(X_test_s)
proba = rf.predict_proba(X_test_s)[:, 1]

print("\n Bot Detection Results:")
print(classification_report(y_test, preds, target_names=['Human', 'Bot']))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

#Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', figsize=(10, 8), title='Feature Importance — Bot Detection')
plt.tight_layout()
plt.savefig('bot_features.png', dpi=100)
print("Feature importance plot saved!")

def check_account(data_dict):
    row = pd.DataFrame([data_dict])
    row['follower_following_ratio'] = row['follower_count'] / (row['following_count'] + 1)
    row['activity_score'] = row['tweets_per_day'] * (1 / (row['account_age_days'] + 1))
    row_scaled = scaler.transform(row)
    pred = rf.predict(row_scaled)[0]
    prob = rf.predict_proba(row_scaled)[0]
    return ("BOT" if pred else "👤 HUMAN"), f"{max(prob)*100:.1f}% confidence"

#Test
sample = {
    'follower_count': 12, 'following_count': 4500, 'tweet_count': 45000,
    'account_age_days': 30, 'tweets_per_day': 150, 'has_profile_pic': 0,
    'has_bio': 0, 'verified': 0, 'avg_retweet_ratio': 0.95,
    'url_in_bio': 1, 'default_profile_theme': 1, 'reply_ratio': 0.01
}
label, conf = check_account(sample)
print(f"\n Sample account check: {label} ({conf})")