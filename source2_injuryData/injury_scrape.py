# For this data-scraping script we are scrapping data from
# https://www.spotrac.com/nba/injured-reserve/
# to get the most recent year's injured players, and then
# saving 

import requests
from bs4 import BeautifulSoup
import csv
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

# URL of the website. We are getting data for the 4 most recent years,
# anything beyond that requires a premium account.
base_url = ''
years = ['2023', '2022', '2021', '2020']

for year in years:
    url = f'{base_url}{year}/'

    if year == '2023':
        # Send a GET request to the URL
        response = requests.get(url)
    else:
        # Send a POST request with appropriate data
        payload = {'sportUrl': 'nba', 'year': year}
        response = requests.post(url, data=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the HTML content
        html_content = response.content

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all tables with class 'datatable'
        tables = soup.find_all('table', class_='datatable')

        # We only need tables 4, 5, 6
        for i, table in enumerate(tables[3:]):
            # Extract data from the target table
            table_data = extract_table_data(table)

            # Clean the data of the table
            for row in table_data[1:]:
                # Drop everything after \n in the first column because that's just redundant data
                row[0] = row[0].split('\n')[0]
                # Swap every non-numeric character in the last two columns with '' because they follow
                # $ddd,ddd,ddd format
                for j in [-1, -2]:
                    row[j] = re.sub('[^0-9]', '', row[j])

            
            # Save the cleaned table to csv, change name depending on table & year
            if i == 0:
                csv_filename = f"injured_reserve_{year}.csv"
            elif i == 1:
                csv_filename = f"rest_reserve_{year}.csv"
            elif i == 2:
                csv_filename = f"personal_reserve_{year}.csv"
            else:
                # sanity check
                print("something wrong with the program/website")
            
            # Opening file & saving it
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row in table_data:
                    csvwriter.writerow(row)
            
            print(f"Table {i + 1} Data for {year} saved to {csv_filename}")
            
    else:
        print(f"Failed to retrieve the webpage for {year}. Status code: {response.status_code}")
