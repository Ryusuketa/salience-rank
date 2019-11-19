import pandas as pd
import glob
import re

from typing import List


def load_single_file(path: str) -> str:
    with open(path, 'r') as f:
        text = f.read()

    text = re.sub('[\n|\t]', ' ', text)
    
    return text


def load_text(file_list: List[str]) -> pd.DataFrame:
    texts = list(map(lambda x: load_single_file(x), file_list))
    file_name = list(map(lambda x: '{:05d}'.format(int(x.split('/')[-1].split('.')[0])), file_list))

    df = pd.DataFrame(dict(text=texts, file_name=file_name)).sort_values('file_name')

    return df


if __name__ == '__main__':
    file_path_list = glob.glob('KeywordExtractor-Datasets-master/datasets/Inspec/docsutf8/*.txt')

    df = load_text(file_path_list)
    print(df.head())