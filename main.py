# This is the main function of the program.
# You only need to run this file.

# Print notification messages and scrape headline data from ESPN
import time
print("\n", "Attempting to scrape headlines from ESPN", "\n")
print("""A chrome browser window will pop up,
please do not do anything to it,
it will close itself once javascripts are all loaded.""", "\n")
time.sleep(2)
from source3_espn.espn_scrape import headlines

print("Here are the most recent NBA headlines in case you missed them:", "\n")
for i, headline in zip(range(len(headlines)), headlines):
    print(f"Headline {i+1}: {headline}")