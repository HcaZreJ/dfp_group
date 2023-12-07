# This is the main function of the program.
# You only need to run this file.

import time
import pandas as pd
from source2_injuryData.injury_scrape import get_injury_data
from source3_espn.espn_scrape import get_headlines

# Print notification messages and scrape headline data from ESPN
print("\nAttempting to scrape headlines from ESPN\n")
print("A chrome browser window will pop up,\nplease do not do anything to it,\nit will close itself once all javascripts are loaded.\n")
time.sleep(2)
headlines = get_headlines()

# Print the most recent headlines to the user
print("Here are the most recent NBA headlines in case you missed them:", "\n")
for i, headline in zip(range(len(headlines)), headlines):
    print(f"Headline {i+1}: {headline}")




# TEST: attempt to read injury data csv
injury_data = pd.read_csv("source2_injuryData/injury_df.csv", header=0, index_col=0)

print(injury_data)