import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.base import clone
import joblib


def save_to_unified_results(title, content):
    """Save output to unified results file"""
    results_file = 'PROJECT_RESULTS.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(results_file, 'a') as f:
        f.write(f'\n{"="*80}\n')
        f.write(f'[{timestamp}] {title}\n')
        f.write(f'{"="*80}\n')
        f.write(str(content))
        f.write('\n')
    
    return results_file


def generate_synthetic_credit_data(n_samples=10000, random_state=42):
    np.random.seed(random_state)

    # Core numeric features
    income = np.random.normal(loc=65000, scale=18000, size=n_samples).clip(12000, 250000)
    debt = np.random.normal(loc=15000, scale=10000, size=n_samples).clip(0, 120000)
    late_payments = np.random.poisson(lam=1.1, size=n_samples)
    credit_utilization = np.random.beta(a=2.7, b=5.2, size=n_samples)  # 0..1
    months_on_book = np.random.randint(6, 240, size=n_samples)
    num_credit_lines = np.random.randint(1, 16, size=n_samples)
    age = np.random.randint(18, 75, size=n_samples)

    # Categorical features for encoding
    loan_purpose = np.random.choice(['debt_consolidation', 'home_improvement', 'medical', 'education', 'other'], size=n_samples, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    employment_status = np.random.choice(['employed', 'self-employed', 'unemployed', 'retired'], size=n_samples, p=[0.6, 0.15, 0.15, 0.1])

    # Generate synthetic target (1 = poor credit, 0 = good credit)
    risk_score = (
        0.00005 * debt
        + 0.7 * (late_payments / 10)
        + 1.2 * credit_utilization
        + 0.000001 * (250000 - income)
        + 0.15 * (age < 24)
        + 0.05 * (employment_status == 'unemployed').astype(float)
        + 0.03 * (employment_status == 'retired').astype(float)
        + 0.3 * (loan_purpose == 'debt_consolidation').astype(float)
    )
    risk_score = (risk_score - risk_score.mean()) / risk_score.std()
    probs = 1 / (1 + np.exp(-risk_score))
    target = (probs > 0.5).astype(int)

    data = pd.DataFrame({
        'income': income,
        'debt': debt,
        'late_payments': late_payments,
        'credit_utilization': credit_utilization,
        'months_on_book': months_on_book,
        'num_credit_lines': num_credit_lines,
        'age': age,
        'loan_purpose': loan_purpose,
        'employment_status': employment_status,
        'label': target,
    })

    # Introduce missing values intentionally for robust cleaning
    for col in ['income', 'debt', 'late_payments', 'loan_purpose']:
        idx = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        data.loc[idx, col] = np.nan

    return data


def build_preprocessing_pipeline():
    numeric_features = ['income', 'debt', 'late_payments', 'credit_utilization', 'months_on_book', 'num_credit_lines', 'age']
    categorical_features = ['loan_purpose', 'employment_status']

    numeric_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transform, numeric_features),
        ('cat', categorical_transform, categorical_features),
    ])

    return preprocessor


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred),
    }


def train_models():
    print('Generating synthetic dataset...')
    df = generate_synthetic_credit_data(n_samples=12000)

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocessor = build_preprocessing_pipeline()

    models = {
        'LogisticRegression': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))]),
        'DecisionTree': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=8, random_state=42))]),
        'RandomForest': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))]),
    }

    results = {}
    best_model_name, best_score = None, -1

    for name, model in models.items():
        print(f'\nTraining {name}...')
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f'{name} metrics: Precision={metrics["precision"]:.4f}, Recall={metrics["recall"]:.4f}, F1={metrics["f1_score"]:.4f}, ROC-AUC={metrics["roc_auc"]:.4f}')
        print(metrics['classification_report'])
        
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model_name = name
    
    # Save training summary to unified results file
    training_summary = f"""
TRAINING COMPLETE
Dataset Size: 12,000 samples
Training Set: 9,000 samples (75%)
Test Set: 3,000 samples (25%)

MODELS TRAINED:
1. Logistic Regression - ROC-AUC: {results['LogisticRegression']['roc_auc']:.4f}
2. Decision Tree - ROC-AUC: {results['DecisionTree']['roc_auc']:.4f}
3. Random Forest - ROC-AUC: {results['RandomForest']['roc_auc']:.4f}

BEST MODEL: {best_model_name} (ROC-AUC: {best_score:.4f})

DETAILED METRICS:
"""
    for model_name, metrics in results.items():
        training_summary += f"\n{model_name}:\n"
        training_summary += f"  Precision: {metrics['precision']:.4f}\n"
        training_summary += f"  Recall: {metrics['recall']:.4f}\n"
        training_summary += f"  F1-Score: {metrics['f1_score']:.4f}\n"
        training_summary += f"  ROC-AUC: {metrics['roc_auc']:.4f}\n"
    
    save_to_unified_results('MODEL TRAINING', training_summary)
    
    best_model = models[best_model_name]
    model_path = 'best_credit_scoring_model.joblib'
    joblib.dump(best_model, model_path)
    print(f'Best model saved to {model_path}')

    export_path = 'credit_scoring_metrics.json'
    pd.DataFrame(results).to_json(export_path, orient='index', indent=2)
    print(f'Evaluation metrics saved to {export_path}')


def predict_user_data():
    model_path = 'best_credit_scoring_model.joblib'
    
    try:
        model = joblib.load(model_path)
        print(f'\nLoaded best model from {model_path}')
    except FileNotFoundError:
        print(f'Error: Model file {model_path} not found. Please train the model first.')
        return
    
    print('\n=== Credit Scoring Prediction ===')
    print('Enter your financial data (press Enter for default values):')
    
    try:
        income = float(input('Income ($) [default 65000]: ') or 65000)
        debt = float(input('Debt ($) [default 15000]: ') or 15000)
        late_payments = int(input('Number of late payments [default 1]: ') or 1)
        credit_utilization = float(input('Credit utilization (0-1) [default 0.5]: ') or 0.5)
        months_on_book = int(input('Months on book [default 60]: ') or 60)
        num_credit_lines = int(input('Number of credit lines [default 5]: ') or 5)
        age = int(input('Age [default 35]: ') or 35)
        loan_purpose = input('Loan purpose (debt_consolidation/home_improvement/medical/education/other) [default debt_consolidation]: ') or 'debt_consolidation'
        employment_status = input('Employment status (employed/self-employed/unemployed/retired) [default employed]: ') or 'employed'
        
        user_data = pd.DataFrame({
            'income': [income],
            'debt': [debt],
            'late_payments': [late_payments],
            'credit_utilization': [credit_utilization],
            'months_on_book': [months_on_book],
            'num_credit_lines': [num_credit_lines],
            'age': [age],
            'loan_purpose': [loan_purpose],
            'employment_status': [employment_status],
        })
        
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0]
        
        risk_label = 'Poor Credit (High Risk)' if prediction == 1 else 'Good Credit (Low Risk)'
        poor_credit_prob = probability[1] * 100
        good_credit_prob = probability[0] * 100
        
        print('\n=== Prediction Results ===')
        print(f'Credit Score Classification: {risk_label}')
        print(f'Probability of Good Credit: {good_credit_prob:.2f}%')
        print(f'Probability of Poor Credit: {poor_credit_prob:.2f}%')
        print(f'\nInput Summary:')
        print(f'  Income: ${income:,.2f}')
        print(f'  Debt: ${debt:,.2f}')
        print(f'  Late Payments: {late_payments}')
        print(f'  Credit Utilization: {credit_utilization:.2%}')
        print(f'  Age: {age}')
        print(f'  Employment: {employment_status}')
        
        # Save results to file
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_file = 'credit_predictions.txt'
        
        with open(output_file, 'a') as f:
            f.write(f'\n{"="*60}\n')
            f.write(f'Prediction Timestamp: {timestamp}\n')
            f.write(f'{"="*60}\n')
            f.write(f'Credit Score Classification: {risk_label}\n')
            f.write(f'Probability of Good Credit: {good_credit_prob:.2f}%\n')
            f.write(f'Probability of Poor Credit: {poor_credit_prob:.2f}%\n')
            f.write(f'\nFinancial Data:\n')
            f.write(f'  Income: ${income:,.2f}\n')
            f.write(f'  Debt: ${debt:,.2f}\n')
            f.write(f'  Late Payments: {late_payments}\n')
            f.write(f'  Credit Utilization: {credit_utilization:.2%}\n')
            f.write(f'  Months on Book: {months_on_book}\n')
            f.write(f'  Number of Credit Lines: {num_credit_lines}\n')
            f.write(f'  Age: {age}\n')
            f.write(f'  Loan Purpose: {loan_purpose}\n')
            f.write(f'  Employment Status: {employment_status}\n')
        
        print(f'\nResults saved to {output_file}')
        
        # Also save to CSV for easy data analysis
        csv_file = 'credit_predictions.csv'
        csv_exists = os.path.exists(csv_file)
        
        prediction_record = pd.DataFrame({
            'timestamp': [timestamp],
            'classification': [risk_label],
            'good_credit_prob': [good_credit_prob],
            'poor_credit_prob': [poor_credit_prob],
            'income': [income],
            'debt': [debt],
            'late_payments': [late_payments],
            'credit_utilization': [credit_utilization],
            'months_on_book': [months_on_book],
            'num_credit_lines': [num_credit_lines],
            'age': [age],
            'loan_purpose': [loan_purpose],
            'employment_status': [employment_status],
        })
        
        with open(csv_file, 'a') as f:
            prediction_record.to_csv(f, mode='a', header=not csv_exists, index=False)
        
        print(f'Results also saved to {csv_file}')
        
        # Save prediction to unified results file
        prediction_summary = f"""PREDICTION RECORD
Classification: {risk_label}
Good Credit Probability: {good_credit_prob:.2f}%
Poor Credit Probability: {poor_credit_prob:.2f}%

INPUT DATA:
  Income: ${income:,.2f}
  Debt: ${debt:,.2f}
  Late Payments: {late_payments}
  Credit Utilization: {credit_utilization:.2%}
  Months on Book: {months_on_book}
  Number of Credit Lines: {num_credit_lines}
  Age: {age}
  Loan Purpose: {loan_purpose}
  Employment Status: {employment_status}
"""
        save_to_unified_results('CREDIT PREDICTION', prediction_summary)
        
    except ValueError as e:
        print(f'Invalid input: {e}. Please enter valid numbers.')
    except Exception as e:
        print(f'Error during prediction: {e}')


def main():
    while True:
        print('\n=== Credit Scoring Model ===')
        print('1. Train models')
        print('2. Predict credit score for user data')
        print('3. Exit')
        
        choice = input('Enter your choice (1/2/3): ').strip()
        
        if choice == '1':
            train_models()
        elif choice == '2':
            predict_user_data()
        elif choice == '3':
            print('Exiting...')
            break
        else:
            print('Invalid choice. Please enter 1, 2, or 3.')


if __name__ == '__main__':
    main()
