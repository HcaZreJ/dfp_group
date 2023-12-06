from selenium import webdriver
from bs4 import BeautifulSoup
import re

url = 'https://www.espn.com/nba/'

# Set up the webdriver (you need to have the appropriate browser driver installed)
driver = webdriver.Chrome()  # For Chrome, you need ChromeDriver: https://sites.google.com/chromium.org/driver/
driver.get(url)

# Wait for the page to load (you might need to adjust the sleep duration)
driver.implicitly_wait(30)

# Get the page source
html_content = driver.page_source

soup = BeautifulSoup(html_content, 'html.parser')

# Pretty print the HTML content
# print(soup.prettify())

# Close the browser window
driver.quit()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Define a regular expression pattern to match the script content
pattern = re.compile(r'<script type="text/javascript">(.*?)</script>', re.DOTALL)

# Search for the javascripts in the HTML content
javascripts = pattern.findall(html_content)

for script in javascripts:
    # from jsbeautifier import beautify
    # print(beautify(match))
    
    # Look for headlines in each of the javascripts because the order
    # of the javascripts might change
    headline_pattern = re.compile(r'"headline":\s*"([^"]+)"')
    headlines = headline_pattern.findall(script)

    # If headlines are found, then this is the javascript that we are
    # looking for. Keep them in the headlines variable so it can be imported
    if headlines:
        print("Headlines scraped from ESPN.")
        break