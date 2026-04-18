import pandas as pd

INPUT_FILE = "jkpcb_final_sorted.csv"
OUTPUT_FILE = "jkpcb_final_cleaned.csv"

# The months you want to remove if the year is 2026
FUTURE_MONTHS = [
    "April", "May", "June", "July", "August", "September", 
    "October", "November", "December"
]

def remove_future_data():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure Year is treated as a string to avoid type mismatch issues (2026 vs "2026")
    df["Year"] = df["Year"].astype(str)
    
    original_count = len(df)
    
    # Create a condition (mask) for the rows we WANT TO REMOVE
    # Condition: Year is 2026 AND Month is in our list of future months
    is_future_2026 = (df["Year"] == "2026") & (df["Month"].isin(FUTURE_MONTHS))
    
    # Keep only the rows that DO NOT (~) match the condition
    df_cleaned = df[~is_future_2026]
    
    removed_count = original_count - len(df_cleaned)
    
    # Save the cleaned dataset
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    
    print(f"🧹 Removed {removed_count} future rows for April-December 2026.")
    print(f"✅ Final dataset size: {len(df_cleaned)} rows.")
    print(f"💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    remove_future_data()