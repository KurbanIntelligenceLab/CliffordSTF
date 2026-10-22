"""Update generated tables embedded in their manuscript sections."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / "dd_template"


def update_tables(section, tables):
    path = ROOT / "sections" / f"{section}.tex"
    text = path.read_text()
    for name, table in tables.items():
        start = f"% BEGIN GENERATED TABLE: {name}"
        end = f"% END GENERATED TABLE: {name}"
        pattern = re.compile(re.escape(start) + r"\n.*?\n" + re.escape(end), re.DOTALL)
        if len(pattern.findall(text)) != 1:
            raise ValueError(f"Expected one inline table {name} in {path}")
        replacement = start + "\n" + table.rstrip("\n") + "\n" + end
        text = pattern.sub(lambda match: replacement, text)
    path.write_text(text)
