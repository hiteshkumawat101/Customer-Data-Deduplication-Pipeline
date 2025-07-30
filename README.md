# Customer-Data-Deduplication-Pipeline


This pipeline cleans, standardizes, and deduplicates customer data from two CSV files, handling common issues such as typos, inconsistent formatting, and missing values. It uses fuzzy and phonetic matching, canonicalization of nicknames and addresses, data validation, and conflict resolution to produce a unified, high-quality customer record set.

---

 Requirements

- Python 3.8+
- External Libraries: pandas, rapidfuzz, metaphone,hashlib (pip install pandas rapidfuzz metaphone)
- Input Files: file_a_with_nicknames.csv and file_b_with_nicknames.csv in the same directory as the script (see FILE_A and FILE_B in the code for customization).

-------------------------------------------------------------------------------------------

 Usage

1. Prepare Input Files  
   Place your CSV files – named file_a_with_nicknames.csv and file_b_with_nicknames.csv – in the project directory.  
   Important: These files must contain these columns: first_name, last_name, street, zip, phone, and email.  
   Customize FILE_A, FILE_B, and OUTPUT_DIR in the script if your files or paths differ.

2. Run the Pipeline  
   Execute the main script from the command line:  
   python main.py
   This will load, clean, deduplicate, and output the resulting records.

3. Output  
   - Cleaned Dataset: test.csv contains all dataset that use for analysis.
   - Cleaned Dataset with specific Columns same as input file: merged_consumers.csv contains unique, standardized customer records with a new consumer_id.
   - Removed Duplicates: fuzzy_deleted.txt logs all rows removed during fuzzy deduplication.
   - Canonical Mappings: Nickname.txt records nickname→canonical and address→canonical mappings.
   - Error Log: pipeline_error.log is created if a runtime error occurs.

----------------------------------------------------------------------------------------

 Pipeline Overview

1. Data Loading  
   Loads both CSV files, tracking the source of each record (source_file column).

2. Cleaning & Normalization  
   - Names: Convert to title case, detect and fix reversed names (e.g., “traS” → “Start”), and handle missing values.
   - Streets: Standardize to title case, expand common abbreviations (e.g., “St” → “Street”), and fix reversed streets.
   - Phones: Extract digits, handle extensions, and normalize international format. Missing values set to “Unknown”.
   - Emails: Validate format. Invalid or missing emails set to “Unknown”.
   - ZIPs: Left as-is; missing values left as “Unknown”.

3. Canonicalization  
   - Nicknames: Within each group (defined by email/phone/last_name/zip), the longest first name is chosen as canonical.
   - Addresses: The longest street in each group (by email/phone/first_name/last_name/zip) becomes canonical.
   - Confidence: Each canonicalization is assigned a confidence score based on group size and variability.

4. Conflict Resolution  
   For records with the same email/phone/first_name/last_name/zip, the most complete record (fewest missing fields) is kept.

5. Fuzzy Deduplication  
   Uses rapidfuzz to compare records on first_name, last_name, street, and zip.  
   Blocking (grouping by zip or name initial) reduces the number of comparisons.  
   Rows with ≥90% average similarity across fields are considered duplicates; all but one are removed and logged.

6. Final Output  
   Same consumer = Same ID: As long as their first_name, phone, last_name, and zip don’t change, they’ll always get the same consumer_id.. The output CSV includes source tracking and all processing metadata.

-------------------------------------------------------------------------------------------

 Assumptions

- Input Schema: Files must have first_name, last_name, street, zip, phone, and email columns.
- Data Quality: Emails and phones are mostly parseable but may have formatting issues. First names, emails, and phones missing entirely are set to “Unknown”.
- Canonicalization: The longest variant in a group is considered most correct for both nicknames and addresses.
- Blocking: Fuzzy deduplication is scoped to groups (blocks) for performance.
- Internationalization: Only ASCII/English text is supported. Non-English characters may not be handled reliably.
- Case: Matching is case-insensitive after normalization.
- Output Directory: Test/ is created if it does not exist.

-------------------------------------------------------------------------------------------

 Technical Notes & Limitations

- Performance: Fuzzy deduplication is O(n²) per block, which may be slow for large datasets (>50k rows). Consider additional blocking or sampling for very large inputs.
- Parallelization: The pipeline is single-threaded; for large-scale use, consider parallelization.
- Customization: Blocking, canonicalization, and reversal rules are hard-coded but can be edited in the script.
- Validation: The pipeline does not validate input file schemas automatically. Ensure columns are present.
- Logs: All removed duplicates and canonical mappings are logged. These may contain PII—handle with care.
- Case Handling: Names and streets are converted to title case; abbreviations are expanded.
- Edge Cases: Reversed names/streets are detected and fixed, but extremely malformed input may not be recovered.
- Testing: Unit tests cover core functions—run with pytest main.py. Contribute more tests for edge cases.
- Privacy: The pipeline processes PII. Review logs and output for compliance with your data policies.
- Dependencies: See requirements.txt (or install libraries manually as above).

-------------------------------------------------------------------------------------------

 Example

Input Data:

| first_name | last_name | street     | zip   | phone        | email                     |
|------------|-----------|------------|-------|--------------|---------------------------|
| Bill       | Smith     | 123 St     | 10001 | 123-456-7890 | bill.smith@example.com    |
| William    | Smith     | 123 Street | 10001 | 123-456-7890 | william.smith@example.com |
| Mike       | Johnson   | 456 Ave    | 20002 | 234-567-8901 | mike.johnson@example.com  |

Output Data:

| consumer_id | first_name | last_name | street      | zip   | phone        | email                     | source_file              |
|-------------|------------|-----------|-------------|-------|--------------|---------------------------|--------------------------|
| C0000001    | William    | Smith     | 123 Street  | 10001 | 1234567890   | william.smith@example.com | file_a_with_nicknames.csv|
| C0000002    | Mike       | Johnson   | 456 Avenue  | 20002 | 2345678901   | mike.johnson@example.com  | file_b_with_nicknames.csv|

-------------------------------------------------------------------------------------------

Conclusion
	This pipeline provides a comprehensive solution to customer data deduplication, making it easy to clean, standardize, and de-duplicate large customer datasets. It is designed to be flexible and can handle a variety of data inconsistencies, including nickname variations, typos, and missing values.
	For large datasets, consider optimizing the deduplication step, and always review the mappings for nicknames and addresses to ensure accuracy.
