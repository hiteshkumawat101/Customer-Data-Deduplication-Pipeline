import pandas as pd
import re
import os
import uuid
from rapidfuzz import fuzz
from metaphone import doublemetaphone
from collections import defaultdict
import pytest
import hashlib


# --- File paths (adjust as needed) ---
FILE_A = "file_a_with_nicknames.csv"
FILE_B = "file_b_with_nicknames.csv"
OUTPUT_DIR = "Test"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Blocking Strategies ---
def get_block_key(row, strategy="zip"):
    """Generate a blocking key for fuzzy matching."""
    if strategy == "zip":
        return str(row.get("zip", "Unknown"))
    elif strategy == "last_initial":
        return str(row.get("last_name", "Unknown")[0].upper() if pd.notna(row.get("last_name")) and len(str(row.get("last_name"))) > 0 else "Unknown")
    elif strategy == "first_initial":
        return str(row.get("first_name", "Unknown")[0].upper() if pd.notna(row.get("first_name")) and len(str(row.get("first_name"))) > 0 else "Unknown")
    else:
        raise ValueError("Unknown blocking strategy")


# --- Phonetic Matching for Names ---
def get_name_phonetics(name):
    """Return primary and secondary Double Metaphone codes for a name."""
    if pd.isna(name) or not isinstance(name, str):
        return None, None
    primary, secondary = doublemetaphone(name)
    return primary, secondary


# --- Cleaning, Validation, and Fixing Functions (with confidence) ---
def normalize_phone(phone):
    """Normalize phone number, return (number, extension) with error handling."""
    try:
        phone = str(phone).strip()
        if not phone:
            return "", ""
        phone_clean = re.sub(r'[^\dxt+]', '', phone.lower())
        match = re.match(r'(?P<number>[\+\d]+)?(?:x|ext)?(?P<ext>\d*)', phone_clean)
        if match:
            number = match.group('number') or ""
            ext = match.group('ext') or ""
        else:
            number = phone_clean
            ext = ""
        if number.startswith("+"):
            number_std = "+" + re.sub(r'\D', '', number[1:])
        else:
            number_std = re.sub(r'\D', '', number)
        return number_std, ext
    except Exception as e:
        print(f"Phone normalization error: {e}")
        return "", ""


def validate_email(email):
    """Validate email format, with error handling."""
    try:
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", str(email)))
    except:
        return False


def fix_reversed_person_name(name, known_names_set):
    """Fix reversed names, return (corrected_name, confidence) with error handling."""
    try:
        if not name or not isinstance(name, str):
            return name, 0.0
        name = name.strip()
        if name and name[-1].isupper() and name[:-1].islower():
            reversed_name = name[::-1].capitalize()
            if reversed_name.lower() in known_names_set:
                return reversed_name, 1.0
            else:
                return reversed_name, 0.8  # Partial confidence if not in known set
        return name, 0.0
    except:
        return name, 0.0


def fix_reversed_street(street, known_streets_set):
    """Fix reversed streets, return (corrected_street, confidence) with error handling."""
    try:
        if not street or not isinstance(street, str):
            return street, 0.0
        street = street.strip()
        words = street.split()
        reversed_pattern = False
        for w in words[:-1]:
            if len(w) > 1 and w[-1].isupper() and w[:-1].islower():
                reversed_pattern = True
                break
        if reversed_pattern:
            reversed_words = [w[::-1] for w in words[::-1]]
            fixed_street = ' '.join(reversed_words)
            fixed_street_norm = fixed_street.lower()
            for known in known_streets_set:
                if fixed_street_norm == known:
                    return known.title(), 1.0
            return fixed_street, 0.8
        street_norm = street.lower()
        for known in known_streets_set:
            if street_norm == known:
                return street, 0.9
        return street, 0.0
    except Exception as e:
        print(f"Street reversal error: {e}")
        return street, 0.0


# --- Canonicalization Functions (with confidence) ---
def replace_nicknames_with_canonical(df, first_name_col='first_name'):
    """Replace nicknames with canonical names in groups, track confidence."""
    df = df.copy()
    df['first_name_norm'] = df[first_name_col].str.lower().str.strip()
    df['first_name_phonetic'] = df[first_name_col].apply(lambda x: get_name_phonetics(x)[0])
    nickname_dict = {}
    confidence_dict = defaultdict(float)

    # Split DataFrame into rows with and without email
    df_with_email = df[df['email'].notnull() & (df['email'].str.strip() != "")]
    df_without_email = df[df['email'].isnull() | (df['email'].str.strip() == "")]

    # Group with email
    group_cols_with_email = ['email', 'phone_number', 'last_name', 'zip']
    for _, group in df_with_email.groupby(group_cols_with_email):
        names = group['first_name_norm'].unique()
        if len(names) > 1:
            canonical = max(names, key=len)
            for idx, row in group.iterrows():
                if row['first_name_norm'] != canonical:
                    nickname_dict[row['first_name_norm']] = canonical
                    confidence_dict[idx] = len(group) / (len(names) * len(group))
        # If only one name, confidence is 1.0
        for idx, _ in group.iterrows():
            confidence_dict[idx] = 1.0 if len(names) == 1 else confidence_dict.get(idx, 0.0)

    # Group without email
    group_cols_without_email = ['phone_number', 'last_name', 'zip']
    for _, group in df_without_email.groupby(group_cols_without_email):
        names = group['first_name_norm'].unique()
        if len(names) > 1:
            canonical = max(names, key=len)
            for idx, row in group.iterrows():
                if row['first_name_norm'] != canonical:
                    nickname_dict[row['first_name_norm']] = canonical
                    confidence_dict[idx] = len(group) / (len(names) * len(group))
        for idx, _ in group.iterrows():
            confidence_dict[idx] = 1.0 if len(names) == 1 else confidence_dict.get(idx, 0.0)

    df[first_name_col] = df['first_name_norm'].apply(lambda x: nickname_dict.get(x, x)).str.title()
    df.drop(columns=['first_name_norm'], inplace=True)
    df['nickname_confidence'] = df.index.map(confidence_dict)
    return df, nickname_dict, confidence_dict


def replace_address_with_canonical(df, address_col='street'):
    """Replace address variants with canonical in groups, track confidence."""
    df = df.copy()
    df['address_norm'] = df[address_col].str.lower().str.strip()
    address_dict = {}
    confidence_dict = defaultdict(float)
    group_cols = ['email', 'phone_number', 'first_name', 'last_name', 'zip']

    for _, group in df.groupby(group_cols):
        addresses = group['address_norm'].unique()
        if len(addresses) > 1:
            canonical = max(addresses, key=len)
            for idx, row in group.iterrows():
                if row['address_norm'] != canonical:
                    address_dict[row['address_norm']] = canonical
                    confidence_dict[idx] = len(group) / (len(addresses) * len(group))
        for idx, _ in group.iterrows():
            confidence_dict[idx] = 1.0 if len(addresses) == 1 else confidence_dict.get(idx, 0.0)

    df[address_col] = df['address_norm'].apply(lambda x: address_dict.get(x, x)).str.title()
    df.drop(columns=['address_norm'], inplace=True)
    df['address_confidence'] = df.index.map(confidence_dict)
    return df, address_dict, confidence_dict


# --- Source Tracking ---
def load_data_with_source(file_a, file_b):
    """Load CSVs and track source file."""
    df_a = pd.read_csv(file_a).assign(source_file=os.path.basename(file_a))
    df_b = pd.read_csv(file_b).assign(source_file=os.path.basename(file_b))
    return pd.concat([df_a, df_b], ignore_index=True)


# --- Conflict Resolution ---
def resolve_conflicts(group, conflict_fields=['first_name', 'last_name', 'street', 'zip']):
    """Resolve conflicts within a group. Favor most recent/complete record."""
    def completeness_score(row):
        return sum(1 for col in conflict_fields if pd.notna(row[col]))
    group['completeness'] = group.apply(completeness_score, axis=1)
    return group.sort_values(['completeness'], ascending=False).iloc[0]


# --- Fuzzy Deduplication with Blocking and Confidence ---
def fuzzy_deduplicate(df, columns, threshold=90, log_file="fuzzy_deleted.txt", add_confidence=True, block_strategy="zip"):
    """Remove rows that are fuzzy duplicates, using blocking, and track confidence."""
    to_delete = set()
    kept_scores = {}
    deleted_rows = []
    blocks = df.groupby(lambda x: get_block_key(df.loc[x], block_strategy))

    if add_confidence:
        df['fuzzy_confidence'] = 0.0
        df['match_block'] = None

    for _, idx_list in blocks.groups.items():
        block_df = df.loc[list(idx_list)]
        for i in idx_list:
            if i in to_delete:
                continue
            row_i = df.loc[i]
            for j in idx_list:
                if j <= i or j in to_delete:
                    continue
                row_j = df.loc[j]
                sim = sum(fuzz.ratio(str(row_i[col]), str(row_j[col])) for col in columns)
                sim_avg = sim / len(columns)
                if sim_avg >= threshold:
                    to_delete.add(j)
                    deleted_rows.append(row_j.to_dict())
                    if add_confidence and sim_avg > kept_scores.get(i, 0):
                        kept_scores[i] = sim_avg
                        df.at[i, 'match_block'] = get_block_key(df.loc[i], block_strategy)

    for i, confidence in kept_scores.items():
        df.at[i, 'fuzzy_confidence'] = confidence

    with open(log_file, "w", encoding="utf-8") as f:
        for row in deleted_rows:
            f.write(str(row) + "\n")

    df = df.drop(to_delete) if to_delete else df
    return df


# --- Main Pipeline ---
def run_pipeline():
    """Run the enhanced deduplication pipeline."""
    df_test = load_data_with_source(FILE_A, FILE_B)

    # Phonetic matching
    df_test['first_name_phonetic'] = df_test['first_name'].apply(lambda x: get_name_phonetics(x)[0])
    df_test['last_name_phonetic'] = df_test['last_name'].apply(lambda x: get_name_phonetics(x)[0])

    # Build known sets for reversal fixing
    df_test['first_name'] = df_test['first_name'].fillna('Unknown')
    df_test['last_name'] = df_test['last_name'].fillna('Unknown')
    df_test['street'] = df_test['street'].fillna('Unknown')
    known_first_names = set(df_test['first_name'].str.lower().dropna())
    known_last_names = set(df_test['last_name'].str.lower().dropna())
    known_streets_set = set(df_test['street'].str.lower().dropna())

    # Apply reversal fixes with confidence
    df_test['first_name'], df_test['name_reversal_confidence'] = \
        zip(*df_test['first_name'].apply(lambda x: fix_reversed_person_name(x, known_first_names)))
    df_test['last_name'], df_test['lastname_reversal_confidence'] = \
        zip(*df_test['last_name'].apply(lambda x: fix_reversed_person_name(x, known_last_names)))
    df_test['street'], df_test['street_reversal_confidence'] = \
        zip(*df_test['street'].apply(lambda x: fix_reversed_street(x, known_streets_set)))

    # Common street abbreviations
    df_test['street'] = df_test['street'].replace({'St': 'Street', 'Ave': 'Avenue', 'Rd': 'Road'}, regex=True)

    # Rename 'phone' to 'phone_number' if needed
    if 'phone' in df_test.columns and 'phone_number' not in df_test.columns:
        df_test['phone_number'] = df_test['phone']

    # Normalize phones
    df_test['phone_number'] = df_test['phone_number'].apply(lambda x: normalize_phone(x)[0] if pd.notna(x) else 'Unknown')
    df_test['phone_extension'] = df_test['phone'].apply(lambda x: normalize_phone(x)[1] if pd.notna(x) else '')

    # Validate emails
    df_test['email'] = df_test['email'].apply(lambda x: x if validate_email(str(x)) else 'Unknown')

    # Canonicalization with confidence
    df_test, nickname_dict, nickname_confidence = replace_nicknames_with_canonical(df_test)
    df_test, address_dict, address_confidence = replace_address_with_canonical(df_test)

    # Conflict resolution (now optional—typically applied after fuzzy dedupe)
    group_cols = ['email', 'phone_number', 'first_name', 'last_name', 'zip']
    resolved_groups = []
    for _, group in df_test.groupby(group_cols):
        resolved = resolve_conflicts(group)
        resolved_groups.append(resolved)
    df_test = pd.DataFrame(resolved_groups)

    # Fuzzy deduplication with blocking and confidence
    fuzzy_columns = ['first_name', 'last_name', 'street', 'zip']
    df_test = fuzzy_deduplicate(df_test, fuzzy_columns, threshold=90, block_strategy='zip')

    # Final consumer_id

    CORE_FIELDS = ['first_name', 'phone_number', 'last_name', 'zip']
    for field in CORE_FIELDS:
        df_test[field] = df_test[field].fillna('Unknown')

    df_test['hash_input'] = df_test[CORE_FIELDS].astype(str).agg('|'.join, axis=1)
    df_test['consumer_id'] = df_test['hash_input'].apply(
        lambda s: 'C' + hashlib.sha256(s.encode('utf-8')).hexdigest()[:8] )
    
    df_test = df_test.drop(columns=['hash_input'])
    cols = ['consumer_id', 'source_file'] + [c for c in df_test.columns if c not in {'consumer_id', 'source_file'}]
    df_test = df_test[cols]
    df_test.to_csv(f"{OUTPUT_DIR}/full_output.csv", index=False)
    print("Nickname to canonical mapping:\n", nickname_dict)
    print("\nAddress variant to canonical mapping:\n", address_dict)

    FINAL_COLUMNS = [
    'consumer_id',
    'first_name',
    'last_name',
    'street',
    'city',
    'state',
    'zip',
    'email',
    'phone'
]
    df_final=df_test[FINAL_COLUMNS]
    # Save output
    df_final.to_csv(f"{OUTPUT_DIR}/merged_consumers.csv", index=False)

    # Log reversal/confidence stats
    with open(f"{OUTPUT_DIR}/Nickname.txt", "w", encoding="utf-8") as f:
        f.write("Nickname to canonical mapping:\n" + str(nickname_dict) + "\n")
        f.write("Address variant to canonical mapping:\n" + str(address_dict) + "\n")


# --- Error Handling & Logging ---
if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        with open(f"{OUTPUT_DIR}/pipeline_error.log", "w", encoding="utf-8") as f:
            f.write(f"Pipeline error:\n{str(e)}\n")


# --- ------------------------------Unit Tests  ----------------------------
def test_block_key():
    df = pd.DataFrame({"zip": ["10001", None, "20002"], "last_name": ["Smith", None, "Zhang"]})
    assert get_block_key(df.loc[0], "zip") == "10001"
    assert get_block_key(df.loc[2], "last_initial") == "Z"

def test_phone_normalization():
    assert normalize_phone("+1 (234) 567-8900") == ("+12345678900", "")
    assert normalize_phone("123-4567 x123") == ("1234567", "123")
    assert normalize_phone(None) == ("", "")

def test_name_phonetics():
    assert get_name_phonetics("Smith") == ("SM0", "XMT")
    assert get_name_phonetics(None) == (None, None)

def test_reversed_name_fix():
    known = {"john"}
    assert fix_reversed_person_name("nhoJ", known)[1] == 1.0
    assert fix_reversed_person_name("nhoX", known)[1] == 0.8
    assert fix_reversed_person_name("nhoX", {})[1] == 0.8


def test_fix_reversed_street():
    known = {"main street"}
    assert fix_reversed_street("traS niaM", known)[1] == 0.8
    # assert fix_reversed_street("traS niab", known) == ("bain Street", 0.8)
    assert fix_reversed_street("Main St", known)[1] == 0.0
    assert fix_reversed_street(None, known) == (None, 0.0)

def test_validate_email():
    assert validate_email("good@example.com") is True
    assert validate_email("bad@") is False
    assert validate_email(None) is False

# To run tests: pytest main.py
