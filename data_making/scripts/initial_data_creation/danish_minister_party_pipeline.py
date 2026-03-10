"""
Danish Minister → Party Letter Dictionary & Assignment Pipeline
================================================================
Covers all ministers and prime ministers in Denmark from 2018 onwards,
spanning three cabinets:

  • Lars Løkke Rasmussen III  (2016–2019)   — VLAK government
  • Frederiksen I             (2019–2022)   — Social Democratic minority
  • Frederiksen II            (2022–present)— SVM government

Party letters follow the official Danish Folketing convention:
  S  – Socialdemokratiet            (Social Democrats)
  KF  – Det Konservative Folkeparti  (Conservative People's Party)
  LA  – Liberal Alliance
  M  – Moderaterne                  (The Moderates)
  V  – Venstre                      (Liberal Party of Denmark)

Design
------
* MINISTER_HISTORY  maps each name to a list of (party, valid_from, valid_until)
  tuples, ordered chronologically. Pass None for open-ended boundaries.
* assign_party()  skips rows where the party column is already filled
  (non-empty string and non-NA). It only writes to rows that are empty / NA.
* Date-aware lookup: the date in the row is used to select the correct party
  for ministers whose affiliation changed over time (e.g. Lars Løkke Rasmussen).
"""

import pandas as pd
from datetime import date, datetime
from typing import Optional
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Affiliation = tuple[str, Optional[date], Optional[date]]


def _d(iso: str) -> date:
    return datetime.strptime(iso, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# 1.  MINISTER HISTORY  (date-aware)
# ---------------------------------------------------------------------------
# Each entry: list of (party_letter, valid_from, valid_until)
# None = unbounded on that side

MINISTER_HISTORY: dict[str, list[Affiliation]] = {

    # ── Cross-cabinet / party-changers (date-sensitive) ──────────────────────

    "Lars Løkke Rasmussen": [
        ("V", None,             _d("2022-12-14")),   # PM in VLAK; out of government 2019–2022
        ("M", _d("2022-12-15"), None),               # FM in Frederiksen II
    ],

    # Støjberg was minister only while V (2015–2019), but encode the full
    # timeline so any post-2021 appearance in the data gets Æ.
    "Inger Støjberg": [
        ("V", None,             _d("2021-02-03")),   # Left Venstre 4 Feb 2021
        ("Æ", _d("2021-02-04"), None),               # Founded Danmarksdemokraterne Jun 2022
    ],

    # ── Venstre (V) ───────────────────────────────────────────────────────────
    "Claus Hjort Frederiksen":      [("V", None, None)],
    "Eva Kjer Hansen":              [("V", None, None)],
    "Troels Lund Poulsen":          [("V", None, None)],
    "Kristian Jensen":              [("V", None, None)],
    "Ulla Tørnæs":                  [("V", None, None)],
    "Jakob Ellemann-Jensen":        [("V", None, None)],
    "Ellen Trane Nørby":            [("V", None, None)],
    "Lars Christian Lilleholt":     [("V", None, None)],
    "Tommy Ahlers":                 [("V", None, None)],
    "Karsten Lauritzen":            [("V", None, None)],   # Tax minister 2015–2019
    "Sophie Løhde":                 [("V", None, None)],
    "Jacob Jensen":                 [("V", None, None)],   # NB: "Jacob" not "Jens"
    "Thomas Danielsen":             [("V", None, None)],
    "Stephanie Lose":               [("V", None, None)],
    "Louise Schack Elholm":         [("V", None, None)],
    "Morten Dahlin":                [("V", None, None)],
    "Torsten Schack Pedersen":      [("V", None, None)],
    "Marie Bjerre":                 [("V", None, None)],
    "Mia Wagner":                   [("V", None, None)],

    # ── Liberal Alliance (LA) ──────────────────────────────────────────────────
    "Anders Samuelsen":             [("LA", None, None)],
    "Simon Emil Ammitzbøll-Bille":  [("LA", None, None)],
    "Ole Birk Olesen":              [("LA", None, None)],
    "Merete Riisager":              [("LA", None, None)],
    "Mette Bock":                   [("LA", None, None)],
    "Thyra Frank":                  [("LA", None, None)],

    # ── Conservative (KF) ─────────────────────────────────────────────────────
    "Søren Pape Poulsen":           [("KF", None, None)],
    "Mai Mercado":                  [("KF", None, None)],
    "Rasmus Jarlov":                [("KF", None, None)],

    # ── Socialdemokratiet (S) ─────────────────────────────────────────────────
    "Mette Frederiksen":            [("S", None, None)],
    "Mattias Tesfaye":              [("S", None, None)],
    "Nicolai Wammen":               [("S", None, None)],
    "Magnus Heunicke":              [("S", None, None)],
    "Ane Halsboe-Jørgensen":        [("S", None, None)],
    "Morten Bødskov":               [("S", None, None)],
    "Kaare Dybvad":                 [("S", None, None)],
    "Kaare Dybvad Bek":             [("S", None, None)],
    "Jeppe Bruus":                  [("S", None, None)],
    "Peter Hummelgaard Thomsen":    [("S", None, None)],
    "Peter Hummelgaard":            [("S", None, None)],
    "Pernille Rosenkrantz-Theil":   [("S", None, None)],
    "Dan Jørgensen":                [("S", None, None)],
    "Sophie Hæstorp Andersen":      [("S", None, None)],
    "Rasmus Stoklund":              [("S", None, None)],
    "Christian Rabjerg Madsen":     [("S", None, None)],
    "Jeppe Kofod":                  [("S", None, None)],
    "Rasmus Prehn":                 [("S", None, None)],
    "Lea Wermelin":                 [("S", None, None)],
    "Benny Engelbrecht":            [("S", None, None)],
    "Nick Hækkerup":                [("S", None, None)],
    "Trine Bramsen":                [("S", None, None)],
    "Jesper Petersen":              [("S", None, None)],
    "Astrid Krag":                  [("S", None, None)],
    "Simon Kollerup":               [("S", None, None)],
    "Flemming Møller Mortensen":    [("S", None, None)],
    "Joy Mogensen":                 [("S", None, None)],
    "Mogens Jensen":                [("S", None, None)],

    # ── Moderaterne (M) ──────────────────────────────────────────────────────
    "Lars Aagaard":                 [("M", None, None)],
    "Jakob Engel-Schmidt":          [("M", None, None)],
    "Mette Kierkgaard":             [("M", None, None)],
    "Christina Egelund":            [("M", None, None)],
    "Caroline Stage Olsen":         [("M", None, None)],
}


# ---------------------------------------------------------------------------
# 2.  DATE-AWARE LOOKUP
# ---------------------------------------------------------------------------

def _parse_row_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return pd.to_datetime(str(value)).date()
    except Exception:
        return None


def lookup_party(
    name: str,
    row_date: Optional[date],
    history: dict[str, list[Affiliation]] = MINISTER_HISTORY,
) -> Optional[str]:
    """
    Return the party letter for *name* at *row_date*.
    Falls back to the most recent affiliation if date is unavailable or
    falls outside all defined windows.
    """
    name = " ".join(name.strip().split())
    records = history.get(name)
    if not records:
        return None

    if row_date is None:
        return records[-1][0]

    for party, valid_from, valid_until in records:
        from_ok = (valid_from is None) or (row_date >= valid_from)
        until_ok = (valid_until is None) or (row_date <= valid_until)
        if from_ok and until_ok:
            return party

    return records[-1][0]


# ---------------------------------------------------------------------------
# 3.  ASSIGNMENT PIPELINE
# ---------------------------------------------------------------------------

def _is_empty(value) -> bool:
    """True if the cell should be treated as 'no party assigned yet'."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def assign_party(
    df: pd.DataFrame,
    speaker_col: str = "speaker",
    date_col: str = "date",
    party_col: str = "party",
    history: dict[str, list[Affiliation]] = MINISTER_HISTORY,
) -> pd.DataFrame:
    """
    Populate *party_col* for rows where it is currently empty / NA.

    Rows that already have a non-empty party value are left UNTOUCHED.
    Party lookup is date-aware: the date in *date_col* selects the correct
    affiliation for ministers whose party changed over time.

    Parameters
    ----------
    df          : Input DataFrame (copy returned; original not modified).
    speaker_col : Column with minister / speaker names.
    date_col    : Column with the date of the speech / record.
    party_col   : Column to write party letters into.
    history     : Name → affiliation history mapping.

    Returns
    -------
    DataFrame with *party_col* filled where a match was found and the
    cell was previously empty.
    """
    df = df.copy()

    if party_col not in df.columns:
        df[party_col] = pd.NA

    matched = skipped_prefilled = unmatched = 0
    unmatched_names: set[str] = set()

    for idx, row in df.iterrows():
        # Skip rows that already have a party assigned
        if not _is_empty(row.get(party_col)):
            skipped_prefilled += 1
            continue

        raw_name = row.get(speaker_col, "")
        if _is_empty(raw_name):
            continue

        name = " ".join(str(raw_name).strip().split())
        row_date = _parse_row_date(row.get(date_col))
        party = lookup_party(name, row_date, history)

        if party:
            df.at[idx, party_col] = party
            matched += 1
        else:
            unmatched += 1
            unmatched_names.add(name)

    total = len(df)
    print(f"[assign_party]  Total rows          : {total}")
    print(f"[assign_party]  Pre-filled (skipped): {skipped_prefilled}")
    print(f"[assign_party]  Newly matched       : {matched}")
    print(f"[assign_party]  Unmatched / empty   : {unmatched}")
    if unmatched_names:
        print(f"[assign_party]  Unmatched names (up to 20):")
        for n in sorted(unmatched_names)[:20]:
            print(f"                  {n!r}")

    return df


# ---------------------------------------------------------------------------
# 4.  QUICK DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='Party Affiliantions',
                    description='Apply missing party affiliantions for ministers')
    
    parser.add_argument("--xml_path", 
                        type=Path,
                        required=True) # add argument
    
    args = parser.parse_args()

    df_xml = pd.read_csv(args.xml_path)

    #make date a datetime object
    df_xml["date"] = pd.to_datetime(df_xml["date"])
    print("\n── Input DataFrame ──────────────────────────────────────────────────")
    print(df_xml.to_string(index=False))

    result_df = assign_party(df_xml)

    print("\n── Output DataFrame ─────────────────────────────────────────────────")
    print(result_df.to_string(index=False))

    out_path = "/mnt/user-data/outputs/demo_output.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nDemo output saved to {out_path}")
