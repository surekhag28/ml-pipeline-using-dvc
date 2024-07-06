import yaml
import json
import logging
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s - %(levelname)s - %(module)s - '
        '%(funcName)s - %(message)s'
    ),
    handlers=[
        logging.FileHandler("model_evaluation.log", mode='w'),
        logging.StreamHandler()
    ]
)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        logger.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error occurred during model evaluation: %s', e)
        raise
    
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise
    
    
def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")
        
if __name__=='__main__':
    main()