import argparse
from pathlib import Path


def extract_molecule(input: Path, output: Path, chain: str, resid: int) -> None:
    atom_lines = []
    conect_lines = []
    selected_serials: set = set()

    with open(input, "r", encoding="utf-8") as f:
        for line in f:
            rec = line[:6].strip()

            if rec in {"ATOM", "HETATM"}:
                chain_id = line[21].strip()
                resid_str = line[22:26].strip()
                serial_str = line[6:11].strip()

                if not resid_str or not serial_str:
                    continue

                if chain_id == chain and int(resid_str) == resid:
                    atom_lines.append(line)
                    selected_serials.add(int(serial_str))

            elif rec == "CONECT":
                conect_lines.append(line)

    filtered_conect = []
    for line in conect_lines:
        fields = line.split()
        if len(fields) < 2:
            continue

        nums = []
        for x in fields[1:]:
            try:
                nums.append(int(x))
            except ValueError:
                pass

        if not nums:
            continue

        src = nums[0]
        dst = nums[1:]

        if src in selected_serials:
            kept = [x for x in dst if x in selected_serials]
            if kept:
                filtered_conect.append(
                    f"CONECT{src:>5}" + "".join(f"{x:>5}" for x in kept) + "\n"
                )

    with open(output, "w", encoding="utf-8") as out:
        out.write("TITLE     Extracted kerogen fragment\n")
        out.write("MODEL        1\n")
        for line in atom_lines:
            out.write(line)
        for line in filtered_conect:
            out.write(line)
        out.write("ENDMDL\nEND\n")

    print(f"Saved {output}")
    print(f"Atoms: {len(atom_lines)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract one kerogen molecule from PDB")
    parser.add_argument("input", type=Path, help="Input PDB file (e.g. Ker.pdb)")
    parser.add_argument("output", type=Path, help="Output PDB file")
    parser.add_argument("--chain", default="A", help="Chain ID (default: A)")
    parser.add_argument("--resid", type=int, default=1, help="Residue ID (default: 1)")
    args = parser.parse_args()

    extract_molecule(args.input, args.output, args.chain, args.resid)
