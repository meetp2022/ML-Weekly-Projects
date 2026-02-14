"""Service for detecting text modality (Prose vs Technical/Code)."""
import re
from typing import Dict

def detect_modality(text: str) -> Dict[str, any]:
    """Detect if the text is standard prose or technical/code.
    
    Returns:
        Dict with modality type and metadata
    """
    # Common code markers
    code_patterns = [
        r'\bimport\b.*\bfrom\b',
        r'\bdef\b\s+\w+\s*\(',
        r'\bclass\b\s+\w+[:\(]',
        r'const\s+\w+\s*=',
        r'let\s+\w+\s*=',
        r'var\s+\w+\s*=',
        r'\bpublic\s+class\b',
        r'Console\.WriteLine',
        r'System\.out\.println',
        r'\s*=\s*\[.*\]',
        r'\s*=\s*\{.*\}',
        r'\bif\b\s*\(.*\)\s*\{',
        r'\bfunction\b\s*\w*\s*\(',
        r'#include\s+<.*>',
        r'<\?php',
        r'pip\s+install',
        r'npm\s+install',
        r'docker-compose',
        r'\.py$',
        r'\.js$',
        r'--\w+',
    ]
    
    # Structural markers
    lines = text.split('\n')
    total_lines = len(lines)
    if total_lines == 0:
        return {"type": "PROSE", "confidence": 0.0}

    indent_count = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
    symbol_count = len(re.findall(r'[{}()\[\]=<>:;]', text))
    word_count = len(text.split())
    
    # Calculate ratios
    code_ratio = 0
    pattern_matches = 0
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            pattern_matches += 1
            
    # Heuristic score
    # High symbol-to-word ratio is a strong indicator of code
    symbol_density = symbol_count / word_count if word_count > 0 else 0
    indent_density = indent_count / total_lines if total_lines > 0 else 0
    
    is_technical = (
        pattern_matches >= 2 or 
        (pattern_matches >= 1 and symbol_density > 0.3) or
        symbol_density > 0.5 or
        (indent_density > 0.4 and symbol_density > 0.2)
    )
    
    if is_technical:
        return {
            "type": "TECHNICAL",
            "confidence": min(0.5 + (pattern_matches * 0.1) + (symbol_density * 0.2), 1.0),
            "reason": "Code-like structures or syntax detected"
        }
    
    return {
        "type": "PROSE",
        "confidence": 1.0,
        "reason": "Natural language patterns detected"
    }
