import time
import pandas as pd
from pathlib import Path
import fcntl  # For file locking on Unix systems

def safe_read_csv(csv_path: Path, max_retries: int = 3, retry_delay: float = 1.0):
    """Safely read CSV file with retry logic to handle concurrent writes."""
    for attempt in range(max_retries):
        try:
            # Try to read the file
            df = pd.read_csv(csv_path)
            return df
        except (pd.errors.EmptyDataError, FileNotFoundError):
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise
        except Exception as e:
            if "truncated" in str(e).lower() or "unexpected end" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"File appears to be being written to, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise
            else:
                raise
    
    return pd.DataFrame()  # Return empty DataFrame if all retries failed 