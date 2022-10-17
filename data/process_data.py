import sys
sys.path.append('data')
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load dataset message and catgories into 1 dataframe
    
    Params:
        - messages_filepath (str): file path of the message csv data file
        - categories_filepath (str): file path of the categories csv data file
    Return:
        a Pandas DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # concat messages to categories
    df = messages.merge(categories, on='id')
    
    return df
    
def clean_data(df):
    '''
    Separate Categories column to 36 columns with binary value and clean the duplicated records 
    
    Params:
        - df (Pandas.DataFrame): raw dataset
    Return:
        cleaned dataframe
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = [col.split('-')[0] for col in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert value for each column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset=['id'], keep=False, inplace=True)
    
    # drop the outlier of storm where row[storm] = 2
    df = df[df.storm != 2]
    
    return df


def save_data(df, database_filename):
    '''
    Save dataframe into Sqlite3 database
    
    Params: 
        - df (Pandas.DataFrame): dataframe want to save
        - database_filename (str): path of .db file where to save dataframe into
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()