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

from rdkit import Chem
from rdkit.Chem import Descriptors
import mordred


def RDKitDescriptors(SMILES, mol):
    """
    Generates the selected subset of RDKit Descriptors
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith("[H]/[N")) # due to a bug of RDKit, a selection the stereoperception is needed
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    complexity = Descriptors.BertzCT(mol)
    charge = Chem.GetFormalCharge(mol)
    hd = Descriptors.NumHDonors(mol)
    ha = Descriptors.NumHAcceptors(mol)
    hac = Descriptors.HeavyAtomCount(mol)
    mollogp = Descriptors.MolLogP(mol)
    aromatic_atoms = sum([mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())])
    heavy_atoms = sum([mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in range(mol.GetNumAtoms())])
    aromaticProportion = aromatic_atoms / heavy_atoms
    return [tpsa, mw, complexity, charge, hd, ha, hac, mollogp, aromaticProportion] # a list of numerical values


def MordredDescriptors(mol_list, calc):
    """
    Generates the Mordred Descriptors for a list of molecules
    :param mol_list: the list of RDKit mol objects
    :param calc: the instance of the Mordred Calculator class, calculates specific descriptors.
    """
    descriptor_data = calc.pandas(mol_list)
    # Mordred's dataframe has been used to calculate for multiple data at the same time.
    descriptor_data = descriptor_data.applymap(
        lambda x: None if isinstance(x, (mordred.error.Missing, mordred.error.Error)) else x)
    return descriptor_data # a dataframe that contains numerical values for each SMILES
