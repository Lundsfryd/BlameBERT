import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from lxml import etree

def extract_tekstgruppe_text(speech_segment) -> str:
    """
    Concatenate all text from every <TekstGruppe> inside a <TaleSegment>,
    joining paragraphs with a space.
    """
    paragraphs = []
    for tg in speech_segment.findall(".//TekstGruppe"):
        lines = [char.text or "" for char in tg.findall(".//Char")]
        paragraph = " ".join(t.strip() for t in lines if t.strip())
        if paragraph:
            paragraphs.append(paragraph)
    return " ".join(paragraphs)

def parse_ft_xml(file_path: str | Path) -> pd.DataFrame:
    """
    Parse a single XML file.

    Returns a DataFrame with one row per <Tale> (speech) containing:
        - paragraph_nr    : index of the <Tale> (paragraph) *within* the file
        - date            : date of meeting (YYYY-MM-DD)
        - speaker         : speaker's name (<OratorFirstName> + <OratorLastName>)
        - party           : party letter (<GroupNameShort>)
        - role            : speakers role in Folketinget ("medlem", "minister", "formand"...)
        - text            : all text from the speaker's <TekstGruppe> elements combined
        - source_file     : basename of the source file
    """
    print(f'Processing {file_path}')
    # This parser auto recovers broken XML (embarassing for Folketinget)
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()

    # Date extraction
    date_el = root.find(".//DateOfSitting")
    if date_el is not None and date_el.text:
        try:
            date_str = datetime.fromisoformat(date_el.text).strftime("%Y-%m-%d")
        except ValueError:
            # If date cannot be extracted, falls back to timestamp (HH:MM:SS)
            date_str = date_el.text.split("T")[0]

    source_file = Path(file_path).name
    rows = []

    for paragraph_nr, tale in enumerate(root.findall(".//Tale")):
        # Speaker metadata 
        fn_el = tale.find(".//OratorFirstName")
        ln_el = tale.find(".//OratorLastName")
        pt_el = tale.find(".//GroupNameShort")
        rl_el = tale.find(".//OratorRole")

        if fn_el is not None:
            first_name = (fn_el.text or "").strip()
        if ln_el is not None:
            last_name = (ln_el.text or "").strip()
        if pt_el is not None:
            party = (pt_el.text or "").strip()
        if rl_el is not None:
            role = (rl_el.text or "").strip()

        speaker = first_name+" "+last_name

        # Text extraction
        tale_segment = tale.find("TaleSegment")
        text = extract_tekstgruppe_text(tale_segment) if tale_segment is not None else ""

        rows.append(
            {
                "paragraph_nr": paragraph_nr,
                "date": date_str,
                "speaker": speaker,
                "party": party,
                "role": role,
                "text": text,
                "source_file": source_file,
            }
        )
    return pd.DataFrame(rows)

def parse_multiple_ft_xml(file_paths: list[str | Path]) -> pd.DataFrame:
    """
    Parse multiple XML files and return a single combined DataFrame. 
    Rows from different files are stacked so "paragraph_nr" resets per file. 
    This way it always reflects the speech's position within its own meeting.

    Parameters
    ----------
    file_paths : list of str or Path

    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns:
        paragraph_nr | date | speaker | party | role | text | source_file
    """
    frames = [parse_ft_xml(fp) for fp in file_paths]
    if not frames:
        return pd.DataFrame(
            columns=["paragraph_nr", "date", "speaker",
                     "party", "role", "text", "source_file"]
        )
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    
    # Works, ideally I want to implement argparse for this instead of hard-coding
    input_path = os.path.join("..","..","..","..","data","xml_meetings")
    folders_in_dir = os.listdir(input_path)
    folder_paths = [os.path.join(input_path, folder) for folder in folders_in_dir]
    files = [os.path.join(folder_path, file) for folder_path in folder_paths for file in os.listdir(folder_path)]
    df = parse_multiple_ft_xml(files)

    # Excluding irrelevant data and creating columns for further processing
    df["chair"] = df["role"] == "formand"
    df = df[df["party"] != "MødeSlut"] # Exclude "MødeSlut" indicators (no data)
    
    # Saving, creating directory if nonexistant
    outpath = os.path.join("..","..","..","..","data","csv_meetings")
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(f'{outpath}/meetings.csv', index=False)
    print(f'Processing finished! Saved to {outpath}/meetings.csv')