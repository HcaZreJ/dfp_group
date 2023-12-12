# For this data-scraping script we defined a function: get_injury_data()
# which can scrape data from https://www.spotrac.com/nba/injured-reserve/
# to get the most recent year's injured players, and save it to a local 
# csv inside the folder. It is used by main.py when the user wants freshly
# scraped data.

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Function to extract data from a table
def extract_table_data(table):
    data = []

    # Extract header row (if thead exists)
    for row in table.find_all(['thead', 'tbody']):
        for tr in row.find_all('tr'):
            row_data = [td.text.strip() for td in tr.find_all('td') + tr.find_all('th')]
            data.append(row_data)

    return data

def get_injury_data():
    """
    Call this function if you need to update the injury_df.csv file inside
    the source 2 folder to get the most up to date data. It will scrape from
    https://www.spotrac.com/nba/injured-reserve/ and update the local csv file.
    """
    # URL of the website. It default loads the data for the most
    # recent year, which is what we want.
    url = 'https://www.spotrac.com/nba/injured-reserve/'
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the HTML content
        html_content = response.content

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all tables with class 'datatable'
        tables = soup.find_all('table', class_='datatable')

        # Create a list to hold all the datatable dataframes
        dfs = []

        # We only need tables 4, 5, 6 (which are for individual players & their injured games)
        for i, table in enumerate(tables[3:]):
            # Extract data from the target table
            table_data = extract_table_data(table)

            # Clean the header of the table of non-letter items
            table_data[0] = [re.sub('[^A-Za-z]', '', heading) for heading in table_data[0]]

            # Change the last Column's name to CashEarnedWhileInjured/Resting so it's consistent
            # across dataframes
            table_data[0][-1] = 'CashEarnedWhileInjured/Resting'

            # Drop the last row because it's an aggregate row
            table_data = table_data[:-1]

            # Clean the data of the table
            for row in table_data[1:]:
                # Drop everything after \n in the first column because that's just redundant data
                row[0] = row[0].split('\n')[0]
                # Swap every non-numeric character in the last two columns with '' because they follow
                # $ddd,ddd,ddd format
                for j in [-1, -2]:
                    row[j] = re.sub('[^0-9]', '', row[j])

            # Turn the table into a dataframe and add it to the holder list
            dfs.append(pd.DataFrame(table_data[1:], columns=table_data[0]))
            
        # Combine the dataframes
        df = pd.concat(dfs)

        # Group by the 'Player' column and aggregate the values
        injury_df = df.groupby('Player').agg({
            'Pos': 'first',  # Take the first position
            'Team': 'first',  # Take the first team
            'Injury': '/'.join,  # Join the injury reasons with a slash
            'Games': 'sum',  # Sum the games
            'CashEarnedWhileInjured/Resting': 'sum',  # Sum the cash earned while injured/Resting
        }).reset_index()

        # Save the result to a local csv
        injury_df.to_csv("source2_injuryData/injury_df.csv")

        # Print a success message
        print("Injury data is scraped and saved to local csv.", "\n")
        
    else:
        print(f"Failed to retrieve the webpage for injury data. Status code: {response.status_code}")
