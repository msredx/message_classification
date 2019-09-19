import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge
    df = messages.merge(categories, on = 'id')
    return df
    


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # get list of new column names
    row = categories.iloc[1,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    #rename columns
    categories.columns = category_colnames
    #get binary codings only
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # in column related, replace 2 with 1
    categories['related'].replace(2,1, inplace=True)
    #remove "child_alone" column because there is no positive case
    categories.drop(columns=['child_alone'], inplace=True)
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # check number of duplicates
    #convert message and original column to string
    df['original'] = df['original'].astype(str)
    df['message'] = df['message'].astype(str)
    # drop duplicates in 'message'
    #df_dupl_removed = 
    df_dupl = df[df.duplicated(subset='message')]
    print(f"removing {df_dupl.shape[0]} duplicates in columns 'message'")
    df_dupl_removed = df.copy()
    df_dupl_removed.drop(index=df_dupl.index, inplace=True)
    #check: are dupl removed?
    print(df_dupl_removed.shape)
    ## check for remaining duplicates in the originals
    # account for original english messages having a "nan" in original column
    df_dupl2 = df_dupl_removed[df_dupl_removed.duplicated(subset='original') & (df_dupl_removed['original'] != 'nan')]
    print(f"removing {df_dupl2.shape[0]} additional duplicates in column 'original'")
    df_dupl_removed.drop(index=df_dupl2.index, inplace=True)
    #check: are dupl removed?
    print(df_dupl_removed.shape)
    print(f"{df.shape[0]-df_dupl_removed.shape[0]} duplicates removed")
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)
    pass  


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