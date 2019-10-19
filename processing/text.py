"""
Utils for cleaning and processing all text encountered by models.
"""

import re
from bert_serving.client import BertClient

bc = BertClient(check_length=True)

# matches https string
secureMatcher = re.compile("https")
# matches non-alphanumeric, space, or sentence-ending punctuation (dash must be at end)
stripMatcher = re.compile(r'[^0-9a-zA-Z\t\n\s_.?!:;/<>*&^%$#@()"~`+-]')
# matches any sequence of tabs, newlines, spaces, underscores, and dashes
spaceMatcher = re.compile(r'[\t\n\s_.?!:;/<>*&^%$#@()"~`+-]+')


## Functions ##
def clean_text(rawString):
    """
    Cleans rawString by replacing spaceMatcher and tagMatcher with a single
    space, removing non-alpha chars, and lowercasing alpha chars
    """
    # replace stripMatcher with ""
    cleanedString = re.sub(stripMatcher, "", rawString)
    # replace spaceMatcher with " " and strip surround whitespace
    spacedString = re.sub(spaceMatcher, " ", cleanedString).strip()
    # lowercase the alpha chars that remain
    loweredString = spacedString.lower()
    return loweredString

def clean_url(rawUrl):
    """ Cleans url by replacing https with http """
    return re.sub(secureMatcher, "http", rawUrl.strip("\n").lower())

def text_to_cls(cleanText):
    """ Converts cleaned text to bert cls token """
    return bc.encode([cleanText])[0]
