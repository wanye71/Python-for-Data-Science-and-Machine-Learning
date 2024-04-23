
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

```