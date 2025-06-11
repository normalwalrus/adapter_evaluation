"""
Filename: lang_utils.py
Author: Digital Hub
Date: 2025-06-11
Version: 0.1.0
Description: 
    provides language filtering for accurate evaluation of languages
"""

import re

def filter_string(input_string: str, lang: str = "en") -> str:
    ''' Orchestrator for language filtering '''
    if lang == "en":
        
        filtered_string = filter_english(input_string=input_string)

    elif lang == "hi":
        
        filtered_string = filter_hindi(input_string=input_string)
    
    else:
        # Code for other languages

        filtered_string = input_string

    return filtered_string

def filter_english(input_string: str) -> str:
    """
    Filter the string such that comparison for WER and evaluation metrics are accurate.
    Current filters:
        - Lowercase all characters
        - Only keep alphabetic characters and whitespaces

    Parameters:
    input_string (str): The string to be filtered.

    Returns:
    String: The filtered string.
    """
    # Convert both texts to lowercase
    input_string = input_string.lower()
    # Use regular expression to find all alphabetic characters and whitespaces
    filtered_string = re.sub(r"[^a-zA-Z\s]", "", input_string)
    
    return filtered_string

def filter_hindi(input_string: str) -> str:
    """ Filtering hindi """
    # Convert input to lowercase (note: Hindi script doesn't have case)
    # Retain only Hindi characters (Devanagari Unicode range) and spaces
    filtered_string = re.sub(r"[^\u0900-\u097F\s]", "", input_string)
    
    return filtered_string

def filter_thai(input_string: str) -> str:
    """ Filtering Thai """
    # Retain only Thai characters and whitespace
    filtered_string = re.sub(r"[^\u0E00-\u0E7F\s]", "", input_string)
    
    return filtered_string
