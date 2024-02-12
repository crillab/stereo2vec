# BSD 3-Clause License
#
# Copyright (c) 2024, Université d'Artois and CNRS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the Université d'Artois and CNRS nor the
#   names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from rdkit import Chem


def MolChiral2vec(mol, radius):
    """
    Calculates ECFP (Morgan fingerprint) with chirality and returns identifiers of substructures as sentences
    Parses the specific symbols/tokens for the MolChiral2vec approaches 
    :param mol: the RDKit mol object of the molecule
    :param radius: the radius of atom environments considered in the fingerprint
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info, useChirality=True)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])
    return list(alternating_sentence)

def MolStereo2vec(mol, radius):
    """
    Calculates ECFP (Morgan fingerprint) with chirality and returns identifiers of substructures as sentences
    Parses the specific symbols/tokens for the MolStereo2vec approaches 
    :param mol: the RDKit mol object of the molecule
    :param radius: the radius of atom environments considered in the fingerprint
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]  # the list of atom indices
    dict_atoms = {x: {r: None for r in radii} for x in
                  mol_atoms}  # storing atom and neighbor information within the radius
    stereochemicalInfos = [None] * len(mol_atoms)
    ez_bonds = []

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    try:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        chiral_dict = {index: stereo for index, stereo in chiral_centers}
    except RuntimeError as error:
        print("Chiral centers cannot be found:", error)
        chiral_dict = {}


    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOCIS or bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
                ez_bonds.append((atom1_idx, atom2_idx, "Z"))
            else:
                ez_bonds.append((atom1_idx, atom2_idx, "E"))

    for element in info:
        for atom_idx, radius_at in info[element]:
            atom = mol.GetAtomWithIdx(atom_idx)
            stereo = atom.GetChiralTag()
            stereochemicalInfo = None
            stereochemicalInfos[atom_idx] = stereochemicalInfo
            dict_atoms[atom_idx][radius_at] = (element, stereochemicalInfo)

            if radius_at == 0:
                if atom_idx in chiral_dict:
                    stereochemicalInfo = chiral_dict[atom_idx]
                elif atom.HasProp("_CIPCode"):
                    stereochemicalInfo = atom.GetProp("_CIPCode")
                elif stereo == rdchem.CHI_TETRAHEDRAL_CW and not atom.HasProp("_CIPCode"):
                    stereochemicalInfo = "R"
                elif stereo == rdchem.CHI_TETRAHEDRAL_CCW and not atom.HasProp("_CIPCode"):
                    stereochemicalInfo = "S"
                stereochemicalInfos[atom_idx] = stereochemicalInfo
                dict_atoms[atom_idx][radius_at] = (element, stereochemicalInfo)
                if ez_bonds != []:
                    for i in range(len(ez_bonds)):
                        stereochemicalInfos[ez_bonds[i][0]] = ez_bonds[i][2]
                        stereochemicalInfos[ez_bonds[i][1]] = ez_bonds[i][2]
                        dict_atoms[ez_bonds[i][0]][radius_at] = (element, ez_bonds[i][2])
                        dict_atoms[ez_bonds[i][1]][radius_at] = (element, ez_bonds[i][2])

    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            if dict_atoms[atom][r]:
                identifier = dict_atoms[atom][r][0]
                stereochemicalInfo = dict_atoms[atom][r][1]
                if stereochemicalInfo == "?":
                    stereochemicalInfo = None
                identifier_tuple = (identifier, stereochemicalInfo)
                identifiers_alt.append(identifier_tuple)
    return list(identifiers_alt)




def IsoSymbol2vec(SMILES):
    """
    Parses the specific symbols/tokens for the IsoSymbol2vec and IsoOrder2vec approaches 
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    """
    smiles_list = []
    i = 0
    while i < len(SMILES):
        if SMILES[i].isalpha():  # for an atom symbol or a charge value or atom repetition value
            atom = SMILES[i]
            i += 1
            while i < len(SMILES) and SMILES[i].islower():
                atom += SMILES[i]
                i += 1
            smiles_list.append(atom)
        elif SMILES[i:i + 2] == "@@":  # for @@ value
            smiles_list.append(SMILES[i:i + 2])
            i += 2
        else:  # for all other values such as "@", "/", "\", "[" ...
            smiles_list.append(SMILES[i])
            i += 1
    return smiles_list


