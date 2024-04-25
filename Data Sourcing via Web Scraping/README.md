
# Python for Data Science and Machine Learning - Data Sourcing via Web Scraping

## Table of Contents

1. [Python requests for automating data collection](#python-requests-for-automating-data-collection)
2. [BeautifulSoup object](#beautifulsoup-object)
3. [NavigableString objects](#navigablestring-objects)
4. [Data parsing](#data-parsing)
5. [Web scraping in practice](#web-scraping-in-practice)
6. [Asynchronous scraping](#asynchronous-scraping)

## Python requests for automating data collection
```python
import requests
response = requests.get('https://www.python.org')
response

## Headers
response.headers

## Content Types
response.headers['Content-Type']

## Body/Content
response.content

import difflib
flines = 'Hello. How are you? I am fine'
glines = 'How are you, Wayne? I am doing well.'

d = difflib.Differ()
diff = d.compare(flines, glines)

for line in diff:
    print(line)
```

### BeautifulSoup object
```python

```

### NavigableString objects
```python

```

### Data parsing
```python

```

### Web scraping in practice
```python

```

### Asynchronous scraping
```python
# In the realm of asynchronous, where tasks take flight,
# 'aiohttp' is summoned, for HTTP requests right.

# aiohttp is a library used for making asynchronous HTTP requests, allowing
# for efficient handling of multiple concurrent requests.
import aiohttp

# Amidst the async dance, where coroutines align,
# 'asyncio' joins, a framework so fine.

# asyncio is a library used for managing asynchronous tasks in Python,
# providing a framework for writing asynchronous code using coroutines.
import asyncio

# In the async world, where files are prime,
# 'aiofiles' steps forth, in an asynchronous rhyme.

# aiofiles is a library used for asynchronous file operations in Python,
# enabling efficient reading and writing of files in an asynchronous manner.
import aiofiles

# Amidst the data's realm, where rows and columns shine,
# 'csv' is called, a reader so fine.

# csv is a module in Python's standard library used for reading and writing
# CSV files, providing functionality to handle comma-separated values data.
import csv

# In the realm of text, where patterns intertwine,
# 're' stands strong, for regex design.

# re is a module in Python's standard library used for working with regular
# expressions, providing powerful pattern matching capabilities.
import re

# In the soup of HTML, where tags come alive,
# 'BeautifulSoup' reigns, to parse and derive.

# BeautifulSoup is a Python library used for parsing HTML and XML documents,
# providing tools for extracting data from HTML/XML files and navigating
# through their structure.
from bs4 import BeautifulSoup

#********************************************************#
nest_asyncio.apply()

#********************************************************#
# In the digital realm, where tasks take flight,
# 'fetch' is defined, to fetch data right.
# With HTTP requests, it ventures forth,
# To bring back data from the internet's vast north.

# fetch is an asynchronous function used to make HTTP GET requests
# using the provided aiohttp session and URL. It returns the text content
# of the response if successful, otherwise prints an error message.
async def fetch(session, url):
    try:
        # The 'try' block allows us to execute code that may potentially
        # raise an exception. If an exception occurs within the 'try' block,
        # execution immediately jumps to the 'except' block.
        # The 'async with' statement is used to asynchronously acquire
        # a resource (in this case, an HTTP response from the provided URL)
        # and automatically release it when the block of code is exited.
        async with session.get(url) as response:
            # Inside the 'async with' block, the response object is available
            # for further processing. Here, we await the response text,
            # which asynchronously reads the response content.
            return await response.text()
    except Exception as e:
        # The 'except' block is used to handle exceptions that occur
        # within the 'try' block. Here, we catch all exceptions (denoted
        # by 'Exception') and print an error message with the exception details.
        print("Fetch error:", e)

#********************************************************#
# Amidst the web's labyrinth, where links entwine,
# 'scrape_and_save_links' is defined, a function so fine.
# With HTML parsed, and a soup to savor,
# It plucks out links with a digital flavor.

# scrape_and_save_links is an asynchronous function used to parse
# HTML text using BeautifulSoup and extract all links that start with
# 'http' from the provided text. It returns a list of extracted links.
async def scrape_and_save_links(url, text):
    # Create a BeautifulSoup object to parse the HTML text
    soup = BeautifulSoup(text, 'html.parser')
    
    # Extract all links from the parsed HTML that start with 'http'
    links = [link.get('href') for link in soup.find_all('a', href=True) if link['href'].startswith('http')]
    
    # Return the extracted links
    return links
    
#********************************************************#
# In the async realm, where tasks unite,
# 'scrape' is defined, to scrape data's light.
# With URLs to traverse, and files to scribe,
# It gathers links, in an async vibe.

# scrape is an asynchronous function used to scrape data from multiple URLs
# using aiohttp and aiofiles libraries. It writes the extracted links to a CSV file.
async def scrape(urls, filename):
    # Create an aiohttp ClientSession to make asynchronous HTTP requests
    async with aiohttp.ClientSession() as session:
        # Open the file in write mode using aiofiles, with UTF-8 encoding
        async with aiofiles.open(filename, 'w', encoding='utf-8') as file:
            # Create a CSV writer object to write to the file
            writer = csv.writer(file)
            # Write the header row to the CSV file
            headers = ["Analytics", "Python", "LinkedIn", "Pandas"]
            await writer.writerow(headers)
            
            # Asynchronously gather the results of scraping links from each URL
            results = await asyncio.gather(*[scrape_and_save_links(url, await fetch(session, url)) for url in urls])
            
            # Find the maximum number of links among all results
            max_links = max(len(links) for links in results)
            
            # Iterate over the range of maximum links
            for i in range(max_links):
                # Create a row for the CSV file with links from each result
                row = [links[i] if i < len(links) else "" for links in results]
                # Write the row to the CSV file
                await writer.writerow(row)

        # After writing to the file, open it again to read and print its contents
        
        # async with aiofiles.open(filename, 'r', encoding='utf-8') as file:
        #     # Read the entire file content
        #     content = await file.read()
        #     # Print the content
        #     print(content)

#********************************************************#
# In the async realm, where tasks convene,
# 'main' is defined, to orchestrate the scene.
# With URLs to explore, and files to create,
# It calls 'scrape', in a state so great.

# main is an asynchronous function that serves as the entry point
# for the program. It orchestrates the scraping process by defining
# URLs to scrape and the filename for the CSV output.
async def main():
    # Define a list of URLs to scrape
    urls = ['https://analytics.usa.gov', 'https://python.org', 'https://linkedin.com', 'https://pandas.pydata.org/']
    
    # Define the filename for the CSV output
    filename = 'myLinks.csv'
    
    # Asynchronously call the scrape function to scrape data from the URLs
    await scrape(urls, filename)
    return pd.read_csv(filename).fillna('')


#********************************************************#
# In the async realm, where tasks take flight,
# 'main' is run, with all its might.
# It orchestrates the show, with a masterful hand,
# Bringing async tasks to a grand, final stand.

# Use asyncio.run() to run the 'main' coroutine,
# starting the asynchronous program execution.
asyncio.run(main())



















```