"""
Extract Karnataka mandi prices (all years, all districts, all commodities)
from the Hugging Face Agmarknet CSV files.

How to use:
1. Put all downloaded .csv files from the Hugging Face "Files" tab in  fds/raw_data
2. Put this script in  fds  (next to the raw_data folder).
3. Run:  python extract_karnataka.py
   It reads the big files in small pieces, so 8 GB RAM is enough.
4. Output: karnataka_all_years.csv

Version 2: detects each file's date format. Some source files write dates
day-first (25/03/2015) and others month-first (03/25/2015); reading them all
day-first swapped day and month and left some dates blank.
"""
import glob
import os
import pandas as pd

SRC_FOLDER = "raw_data"   # folder next to this script; or put the full path here
OUT_FILE = "karnataka_all_years.csv"
STATE = "karnataka"       # keeps ALL years, ALL districts, ALL commodities for this state

COLS = ["State", "District", "Market", "Commodity", "Variety", "Grade",
        "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price", "Commodity_Code"]


def parse_dates(raw):
    """Work out whether a file's dates are day-first or month-first, then parse them.

    Evidence comes from dates that can only be read one way: a first number above 12
    means day-first, a second number above 12 means month-first. Dates such as 05/06
    could be either, so they follow the file's majority. ISO dates (2015-03-25) are
    read directly.
    """
    raw = raw.fillna("").str.strip().str.split(" ").str[0]   # drop any time part
    parts = raw.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$")
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    dayfirst_votes = int((a > 12).sum())
    monthfirst_votes = int((b > 12).sum())
    fmt = "month-first" if monthfirst_votes > dayfirst_votes else "day-first"

    if fmt == "day-first":
        day, month = a, b
    else:
        day, month = b, a
    # a row that only makes sense the other way round is read the other way round
    swap = (month > 12) & (day <= 12)
    day, month = day.where(~swap, month), month.where(~swap, day)

    dates = pd.to_datetime(
        pd.DataFrame({"year": pd.to_numeric(parts[2], errors="coerce"),
                      "month": month, "day": day}),
        errors="coerce")
    # ISO style 2015-03-25 and anything else pandas can read on its own
    rest = dates.isna() & (raw != "")
    if rest.any():
        dates[rest] = pd.to_datetime(raw[rest], format="ISO8601", errors="coerce")
    stats = {"format": fmt, "dayfirst_votes": dayfirst_votes,
             "monthfirst_votes": monthfirst_votes,
             "ambiguous": int(((a <= 12) & (b <= 12)).sum()),
             "unreadable": int(dates.isna().sum())}
    return dates, stats


files = sorted(glob.glob(os.path.join(SRC_FOLDER, "**", "*.csv"), recursive=True))
print(f"Found {len(files)} CSV files\n")
if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

kept_total = 0
bad_dates = 0
report = []
for f in files:
    name = os.path.basename(f)
    pieces = []
    try:
        reader = pd.read_csv(f, usecols=lambda c: c.strip() in COLS, dtype=str,
                             chunksize=500_000, on_bad_lines="skip")
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            # keep every row whose state name contains "karnataka" (any case/spacing)
            chunk = chunk[chunk["State"].fillna("").str.lower().str.contains(STATE)]
            if not chunk.empty:
                pieces.append(chunk)
    except Exception as e:
        print(f"  !! could not read {name}: {e}")
        continue
    if not pieces:
        print(f"{name}: no Karnataka rows")
        continue

    # date format is decided per file, using all its Karnataka rows together
    kdf = pd.concat(pieces, ignore_index=True)
    kdf["Arrival_Date"], st = parse_dates(kdf["Arrival_Date"])
    kdf = kdf.reindex(columns=COLS)
    kdf["Source_File"] = name
    kdf.to_csv(OUT_FILE, mode="a", index=False, header=not os.path.exists(OUT_FILE))

    kept_total += len(kdf)
    bad_dates += st["unreadable"]
    report.append({"file": name, "rows": len(kdf), **st})
    print(f"{name}: kept {len(kdf):,} rows | dates {st['format']} "
          f"(votes day-first {st['dayfirst_votes']:,} / month-first {st['monthfirst_votes']:,}) "
          f"| unreadable {st['unreadable']:,}")

print(f"\nDone. {kept_total:,} rows saved to {OUT_FILE}")
pd.DataFrame(report).to_csv("date_format_report.csv", index=False)
print("Per-file date report saved to date_format_report.csv")

if kept_total:
    text_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade", "Source_File"]
    df = pd.read_csv(OUT_FILE, parse_dates=["Arrival_Date"],
                     dtype={c: "category" for c in text_cols})   # category keeps memory low
    # Some source files overlap (e.g. 2004-2007.csv and 2007.csv), so remove repeated rows.
    # Source_File is ignored here, otherwise the same row from two files would look different.
    before = len(df)
    df = df.drop_duplicates(subset=COLS)
    df.to_csv(OUT_FILE, index=False)
    print(f"Removed {before - len(df):,} duplicate rows -> {len(df):,} rows remain")
    print("\nRows per year:")
    print(df["Arrival_Date"].dt.year.dropna().astype(int).value_counts().sort_index().to_string())
    print(f"\nRows with an unreadable date (kept, date blank): {bad_dates:,}")
    print("Date range:", df["Arrival_Date"].min(), "to", df["Arrival_Date"].max())
    today = pd.Timestamp.today().normalize()
    print("Rows dated after today:", int((df["Arrival_Date"] > today).sum()))
    print("\nCommodities:", df["Commodity"].nunique(),
          " Markets:", df["Market"].nunique(), " Districts:", df["District"].nunique())
    print(f"File size: {os.path.getsize(OUT_FILE) / 1e6:,.0f} MB")
