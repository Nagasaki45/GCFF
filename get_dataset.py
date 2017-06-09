"""
Download dataset, extract, and remove zip file.
"""
import os
import urllib.request
import zipfile

filename = 'data.zip'

url = 'http://vips.sci.univr.it/research/fformation/download/data.zip'
response = urllib.request.urlopen(url)
with open(filename, 'wb') as f:
    f.write(response.read())

with zipfile.ZipFile(filename) as f:
    f.extractall('.')

os.remove(filename)
