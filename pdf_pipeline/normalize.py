import re
import unicodedata

def normalize(text):
    if not text:
        return "" 
        
    text = unicodedata.normalize("NFKC", text)


    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()