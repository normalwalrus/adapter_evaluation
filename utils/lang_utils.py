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
    
    elif lang == "th": 
        
        filtered_string = filter_thai(input_string=input_string)
    
    elif lang == "vi": 
        
        filtered_string = filter_vietnamese(input_string=input_string)
    
    elif lang == "bn": 
        
        filtered_string = filter_bengali(input_string=input_string)
    
    elif lang == "fil": 
        
        filtered_string = filter_filipino(input_string=input_string)
    
    elif lang == "ms":
        
        filtered_string = filter_malay(input_string=input_string)
    
    elif lang == "id":
        
        filtered_string = filter_indonesian(input_string=input_string)
    
    elif lang == "zh":
        
        filtered_string = filter_simplified_chinese(input_string=input_string)
    
    elif lang == "ar":
        
        filtered_string = filter_arabic(input_string=input_string)
    
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

def filter_vietnamese(input_string: str) -> str:
    """ Filtering Vietnamese """
    # This regex retains all Vietnamese letters and whitespace
    filtered_string = re.sub(r"[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂÊÔƠƯăêôơư\s]", "", input_string)
    
    return filtered_string

def filter_bengali(input_string: str) -> str:
    """ Filtering Bengali """
    # Keep only Bengali characters and whitespace
    filtered_string = re.sub(r"[^\u0980-\u09FF\s]", "", input_string)
    return filtered_string

def filter_filipino(input_string: str) -> str:
    """ Filtering Filipino """
    # Retain only Filipino letters (including ñ, é) and whitespace
    filtered_string = re.sub(r"[^a-zA-ZñÑéÉ\s]", "", input_string)
    return filtered_string

def filter_arabic(input_string: str) -> str:
    """Filtering Arabic letters and whitespace from input string."""
    # Arabic Unicode ranges + whitespace
    filtered_string = re.sub(r"[^\u0600-\u06FF\s]", "", input_string)
    return filtered_string


def filter_simplified_chinese(input_string: str) -> str:
    """Filtering CJK characters (which includes Simplified Chinese) and whitespace."""
    # CJK Unified Ideographs: \u4e00–\u9fff
    filtered_string = re.sub(r"[^\u4e00-\u9fff\s]", "", input_string)
    return filtered_string

def filter_indonesian(input_string: str) -> str:
    """Filtering for only Indonesian alphabet characters and whitespace."""
    filtered_string = re.sub(r"[^a-zA-Z\s]", "", input_string)
    return filtered_string

def filter_malay(input_string: str) -> str:
    """Filtering for only Malay characters and whitespace."""
    # Includes standard Latin letters + common Malay accents + whitespace
    filtered_string = re.sub(r"[^a-zA-ZéÉèÈâÂîÎôÔ\s]", "", input_string)
    return filtered_string
