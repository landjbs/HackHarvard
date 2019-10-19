"""
Functions:
    -clean_text(): Cleans captions so bert can understand them
        -Replace \t, \n, and multiple spaces with a single space
        -Remove non-alpha characters
        -Lowercase all alpha characters
"""

import re
from unidecode import unidecode
secureMatcher = re.compile(r"https")

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
    # replace accented characters with non-accent representation
    deaccentedString = unidecode(rawString)
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
