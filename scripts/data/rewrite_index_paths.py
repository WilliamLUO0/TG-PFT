import argparse
import csv
import os
import shutil
from typing import Dict


def replace_prefix(s: str, rules: Dict[str, str]) -> str:
    if not s:
        return s
    for old, new in rules.items():
        if s.startswith(old):
            return new + s[len(old):]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input index.csv")
    ap.add_argument("--out_csv", default=None, help="Output CSV path (ignored if --inplace)")
    ap.add_argument("--inplace", action="store_true", help="Rewrite in place (creates .bak)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    ap.add_argument("--dry_run", action="store_true", help="Only report counts, do not write output")
    ap.add_argument("--rule", nargs=2, action="append", required=True, metavar=("OLD_PREFIX", "NEW_PREFIX"),
                    help="Replace OLD_PREFIX with NEW_PREFIX at the start of every cell (repeatable)")
    args = ap.parse_args()

    in_csv = args.in_csv
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    if args.inplace:
        out_csv = in_csv + ".tmp_rewritten"
        bak_csv = in_csv + ".bak"
    else:
        if not args.out_csv:
            raise ValueError("Please provide --out_csv, or use --inplace.")
        out_csv = args.out_csv
        bak_csv = None

    rules: Dict[str, str] = dict(args.rule)

    total_cells = 0
    changed_cells = 0
    changed_rows = 0

    with open(in_csv, "r", encoding=args.encoding, newline="") as f_in:
        reader = csv.reader(f_in)

        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("CSV is empty.")

        if args.dry_run:
            for row in reader:
                row_changed = False
                for cell in row:
                    total_cells += 1
                    new_cell = replace_prefix(cell, rules)
                    if new_cell != cell:
                        changed_cells += 1
                        row_changed = True
                if row_changed:
                    changed_rows += 1
            print(f"[DRY RUN] rows_changed={changed_rows}, cells_changed={changed_cells}, total_cells_scanned={total_cells}")
            return

    with open(in_csv, "r", encoding=args.encoding, newline="") as f_in, \
         open(out_csv, "w", encoding=args.encoding, newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            row_changed = False
            new_row = []
            for cell in row:
                total_cells += 1
                new_cell = replace_prefix(cell, rules)
                if new_cell != cell:
                    changed_cells += 1
                    row_changed = True
                new_row.append(new_cell)
            if row_changed:
                changed_rows += 1
            writer.writerow(new_row)

    if args.inplace:
        if os.path.exists(bak_csv):
            raise RuntimeError(f"Backup file already exists, refusing to overwrite: {bak_csv}")
        shutil.move(in_csv, bak_csv)
        shutil.move(out_csv, in_csv)
        print(f"[OK] In-place rewrite done.")
        print(f"     backup: {bak_csv}")
        print(f"     output: {in_csv}")
    else:
        print(f"[OK] Wrote: {out_csv}")

    print(f"[STATS] rows_changed={changed_rows}, cells_changed={changed_cells}, total_cells_processed={total_cells}")
    print("[RULES]")
    for old, new in rules.items():
        print(f"  {old}  ->  {new}")


if __name__ == "__main__":
    main()
