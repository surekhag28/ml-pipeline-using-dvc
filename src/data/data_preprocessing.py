import os
import re
import nltk
import string
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        logger.debug('converted to lower case')
        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug('stop words removed')
        df['content'] = df['content'].apply(removing_numbers)
        logger.debug('numbers removed')
        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug('punctuations removed')
        df['content'] = df['content'].apply(removing_urls)
        logger.debug('urls')
        df['content'] = df['content'].apply(lemmatization)
        logger.debug('lemmatization performed')
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info('Data loaded properly')
        
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        data_path = Path('./data')/'interim'
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv((Path(data_path)/'train_processed.csv'), index=False)
        test_processed_data.to_csv((Path(data_path)/'test_processed.csv'), index=False)
        
        logger.info('Processed data saved to %s', data_path)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s',e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()