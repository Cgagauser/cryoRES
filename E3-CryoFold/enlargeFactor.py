


factor = (factor1,factor2,factor3)

with open("/ziyingz/Programs/E3-CryoFold/example/5uz7/5uz7.pdb") as fin, open("/ziyingz/Programs/E3-CryoFold/example/5uz7/scaled.pdb", "w") as fout:
    for line in fin:
        if line.startswith(("ATOM  ", "HETATM")):
            x = float(line[30:38]) * factor1
            y = float(line[38:46]) * factor2
            z = float(line[46:54]) * factor3
            fout.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        else:
            fout.write(line)
