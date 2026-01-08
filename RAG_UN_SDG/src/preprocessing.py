import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text.strip()
