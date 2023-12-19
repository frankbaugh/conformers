#!/usr/bin/env python
"""
Purpose: Convert a multi-conformer file formats using OpenEye tools.

Usage:   python convertExtension.py -i [file.sdf] -o [file.mol2]

By:      Victoria T. Lim
Version: Nov 05 2019

"""

import openeye.oechem as oechem
import pint

registry = pint.UnitRegistry()

def main(**kwargs):

    # open input file
    ifs = oechem.oemolistream()
    ifs.SetConfTest(oechem.OEAbsCanonicalConfTest())
    if not ifs.open(args.infile):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % args.infile)

    # open output file
    ofs = oechem.oemolostream()
    if not ofs.open(args.outfile):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % args.outfile)

    # write to output
    for mol in ifs.GetOEMols():
        for conf in mol.GetConfs():
            for dp in oechem.OEGetSDDataPairs(conf):
                if dp.GetTag().startswith('r_mmod'):
                    newValue = pint.Quantity(float(dp.GetValue()), 'kJ/mol').to('kcal/mol').magnitude
                    oechem.OESetSDData(conf, dp.GetTag(), str(newValue))
            oechem.OEWriteConstMolecule(ofs, conf)
    ifs.close()
    ofs.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--infile",
            help="Input molecule file")
    parser.add_argument("-o", "--outfile",
            help="Output molecule file")

    args = parser.parse_args()
    opt = vars(args)

    main(**opt)

