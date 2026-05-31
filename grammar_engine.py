# grammar_engine.py
import os
import json

# ==================================================
# 1. DYNAMIC CONFIGURATION LOADER
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LEXICON = {}
PHRASE_TEMPLATES = {}

def load_config():
    global LEXICON, PHRASE_TEMPLATES
    
    # Load Part-of-Speech Lexicon
    lexicon_path = os.path.join(SCRIPT_DIR, "grammar_lexicon.json")
    if os.path.exists(lexicon_path):
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                LEXICON = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading grammar_lexicon.json: {e}")
            LEXICON = {}
    else:
        LEXICON = {}

    # Load Multi-word Phrase Templates
    templates_path = os.path.join(SCRIPT_DIR, "phrase_templates.json")
    if os.path.exists(templates_path):
        try:
            with open(templates_path, "r", encoding="utf-8") as f:
                raw_templates = json.load(f)
                # Convert comma-separated string keys to set/tuple of words
                PHRASE_TEMPLATES = {
                    tuple(k.split(",")): v for k, v in raw_templates.items()
                }
        except Exception as e:
            print(f"⚠️ Error loading phrase_templates.json: {e}")
            PHRASE_TEMPLATES = {}
    else:
        PHRASE_TEMPLATES = {}

# Initialize configurations on module import
load_config()


# ==================================================
# 2. DYNAMIC PART-OF-SPEECH TAGGER
# ==================================================
def get_pos(word, prev_pos=None):
    """
    Returns the POS tag for a word. If not in LEXICON, uses offline heuristics.
    """
    word_lower = str(word).strip().lower()
    if word_lower in LEXICON:
        return LEXICON[word_lower]
        
    # Heuristics for unrecognized words (highly useful when retraining models)
    # 1. Suffix heuristics
    if any(word_lower.endswith(s) for s in ["ing", "ed", "ate", "ify", "ize"]):
        return "VERB"
    if any(word_lower.endswith(s) for s in ["able", "ible", "ful", "less", "ous", "ish", "ive", "ic", "al"]):
        return "ADJ"
    if word_lower.endswith("ly"):
        return "ADV"
    if any(word_lower.endswith(s) for s in ["tion", "ness", "ity", "ment", "er", "or", "ist"]):
        return "NOUN"
        
    # 2. Context-based guessing
    if prev_pos in ["PRON", "NOUN"]:
        return "VERB"  # Subject followed by unknown word is likely a Verb
    if prev_pos == "VERB_DESIRE":
        return "VERB"  # "want" followed by unknown word is likely a Verb
        
    # Default fallback
    return "NOUN"


# ==================================================
# 3. CORE GRAMMAR PROCESSING ENGINE
# ==================================================
def process_sentence(word_buffer):
    """
    Takes an array of raw recognized words/letters and structures them 
    into a grammatically correct English sentence.
    """
    if not word_buffer:
        return ""
        
    # 1. Reload config in case they updated their JSON maps
    load_config()

    # 2. Clean up tokens (lowercase for template/POS matching)
    tokens = [str(w).strip().lower() for w in word_buffer if w]
    if not tokens:
        return ""
    
    # 3. Check for multi-word phrase matches (Order independent)
    token_set = set(tokens)
    for trigger_words, phrase in PHRASE_TEMPLATES.items():
        if set(trigger_words).issubset(token_set):
            return phrase

    # 4. Syntactic Pipeline
    # Convert tokens to word-POS tuples
    tagged_tokens = []
    prev_pos = None
    for token in tokens:
        pos = get_pos(token, prev_pos)
        tagged_tokens.append((token, pos))
        prev_pos = pos

    processed_words = []
    n = len(tagged_tokens)

    # Pronoun maps
    PRONOUN_SUBJECTS = {
        "me": "I",
        "them": "they",
        "us": "we",
        "him": "he",
        "her": "she"
    }
    PRONOUN_OBJECTS = {
        "i": "me",
        "they": "them",
        "we": "us",
        "he": "him",
        "she": "her"
    }

    i = 0
    while i < n:
        word, pos = tagged_tokens[i]
        
        # Lookahead context
        next_word, next_pos = (tagged_tokens[i+1][0], tagged_tokens[i+1][1]) if i + 1 < n else (None, None)
        prev_word = processed_words[-1] if processed_words else None

        # --- Rule 1: Pronoun Case Correction ---
        if pos == "PRON":
            if i == 0 or (next_pos in ["VERB", "VERB_DESIRE"]):
                word = PRONOUN_SUBJECTS.get(word, word)
            elif prev_word and get_pos(prev_word) in ["VERB", "PREP"]:
                word = PRONOUN_OBJECTS.get(word, word)

        # --- Rule 2: Special "Pain" Handling ---
        # ASL: "me pain" -> English: "I am in pain" or "I have pain"
        if word == "pain":
            if prev_word and prev_word.lower() in ["i", "he", "she", "we", "they", "you", "me"]:
                processed_words.append("am in")
                word = "pain"
            elif prev_word and get_pos(prev_word) == "NOUN":
                processed_words.append("is experiencing")
                word = "pain"

        # --- Rule 3: Copula (am/is/are) Insertion ---
        # If subject (PRON/NOUN) is followed by an adjective (e.g. "they sick" -> "they are sick")
        if pos in ["PRON", "NOUN"] and next_pos == "ADJ":
            processed_words.append(word)
            
            # Select correct auxiliary verb
            subj = word.lower()
            if subj in ["i", "me"]:
                copula = "am"
            elif subj in ["we", "they", "you", "them", "parents"]:
                copula = "are"
            else:
                copula = "is"
            processed_words.append(copula)
            i += 1
            continue

        # --- Rule 4: Desire Verb Infinitive Builder ---
        # If desire verb is followed by another verb (e.g., "want eat" -> "want to eat")
        if pos == "VERB_DESIRE" and next_pos == "VERB":
            processed_words.append(word)
            processed_words.append("to")
            i += 1
            continue

        # --- Rule 5: Article & Preposition Stitching ---
        # A. Wait + PRON/NOUN -> wait for PRON/NOUN (e.g., "wait you" -> "wait for you")
        if word == "wait" and next_pos in ["PRON", "NOUN"]:
            processed_words.append(word)
            processed_words.append("for")
            i += 1
            continue
            
        # B. Verb + Noun -> insert appropriate article/preposition
        if pos == "VERB" and next_pos == "NOUN":
            processed_words.append(word)
            if next_word in ["hospital", "college"]:
                processed_words.append("to the")
            elif next_word == "wheelchair":
                processed_words.append("a")
            else:
                # General article fallback for singular common nouns
                # Avoid prefixing plural nouns ending with 's' (like 'weeks') unless it is 'parents'
                if next_word == "parents":
                    processed_words.append("the")
                elif not next_word.endswith("s"):
                    # Check if next_word starts with a vowel
                    starts_with_vowel = next_word[0].lower() in ['a', 'e', 'i', 'o', 'u'] if next_word else False
                    processed_words.append("an" if starts_with_vowel else "the")
            i += 1
            continue

        # Default: append the word
        processed_words.append(word)
        i += 1

    # Join tokens into a sentence string
    sentence = " ".join(processed_words)
    
    # 5. Clean up spacing and capitalize the sentence
    if sentence:
        # Standard grammatical spelling for specific terms
        sentence = sentence.replace("thank you", "thank you")
        sentence = sentence.replace("please", "please")
        
        # Capitalize the first letter
        sentence = sentence[0].upper() + sentence[1:]
        
        # Ensure sentence ends with correct punctuation
        if not sentence.endswith(('.', '!', '?')):
            sentence += "."
            
    return sentence