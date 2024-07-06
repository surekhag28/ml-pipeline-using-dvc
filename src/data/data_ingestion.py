import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s - %(levelname)s - %(module)s - '
        '%(funcName)s - %(message)s'
    ),
    handlers=[
        logging.FileHandler("data_ingestion.log", mode='w'),
        logging.StreamHandler()
    ]
)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s',e)
        raise
    
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.info('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error ocurred while loading the data: %s',e)
        raise
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])]
        final_df['sentiment'].replace({'happiness':1, 'sadness':0}, inplace=True)
        logger.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
    
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = Path(data_path)/ 'raw'
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv((Path(raw_data_path)/'train.csv'), index=False)
        test_data.to_csv((Path(raw_data_path)/'test.csv'), index=False)
        logger.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the file %s',e)
        raise

    
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        raise
    

if __name__=='__main__':
    main()



