from pathlib import Path
from os.path import join, realpath

path_to_data = "/media/andrey/Samsung_T5/PHD/Kerogen/type1matrix/300K/ch4/"
INPUT = join(path_to_data, "Ker.pdb")
OUTPUT = join(path_to_data, "KRG_chainA_res1.pdb")

TARGET_CHAIN = "A"
TARGET_RESID = 1

atom_lines = []
conect_lines = []
selected_serials = set()

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        rec = line[:6].strip()

        if rec in {"ATOM", "HETATM"}:
            chain = line[21].strip()
            resid_str = line[22:26].strip()
            serial_str = line[6:11].strip()

            if not resid_str or not serial_str:
                continue

            resid = int(resid_str)
            serial = int(serial_str)

            if chain == TARGET_CHAIN and resid == TARGET_RESID:
                atom_lines.append(line)
                selected_serials.add(serial)

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

with open(OUTPUT, "w", encoding="utf-8") as out:
    out.write("TITLE     Extracted kerogen fragment\n")
    out.write("MODEL        1\n")
    for line in atom_lines:
        out.write(line)
    for line in filtered_conect:
        out.write(line)
    out.write("ENDMDL\nEND\n")

print(f"Saved {OUTPUT}")
print(f"Atoms: {len(atom_lines)}")
