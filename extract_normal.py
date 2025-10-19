import pandas as pd

CRASH_CSV = r"E:/coding/3dcnn/project/Crash_Table.csv"   # Accident data
NORMAL_CSV = r"E:/coding/3dcnn/project/Normal_Table.csv"  # Normal data
OUTPUT_CSV = r"E:/coding/3dcnn/project/Final_Table.csv"   # Final merged CSV

def merge_csvs():
    crash_df = pd.read_csv(CRASH_CSV)
    normal_df = pd.read_csv(NORMAL_CSV)

    combined_df = pd.concat([crash_df, normal_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Final combined CSV saved to {OUTPUT_CSV}")
    print(f"Accident videos: {len(crash_df)}, Normal videos: {len(normal_df)}")
    print(f"Total videos: {len(combined_df)}")

if __name__ == "__main__":
    merge_csvs()
