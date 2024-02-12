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
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import MACCSkeys
from rdkit import Chem


def MACCSFP(SMILES, mol):
    """
    Generates the 166-dimensional MACCS Keys Fingerprint for a molecule
    :param SMILES: the SMILES representation of the molecule, needed for the selection the stereoperception
    :param mol: the RDKit mol object of the molecule
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith(
        "[H]/[N"))  # due to a bug of RDKit, a selection the stereoperception is needed
    return [int(bit) for bit in
            MACCSkeys.GenMACCSKeys(mol).ToBitString()]  # the bits converted to a list of numerical values


def RDKitFP(SMILES, mol, dimensions):
    """
    Generates the RDKit Fingerprint for a molecule with a specified dimensions
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param dimensions: the length of the bit vector used to represent the molecule with RDKit Fingerprint, by default 2048
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith(
        "[H]/[N"))  # due to a bug of RDKit, a selection the stereoperception is needed
    return [int(bit) for bit in
            RDKFingerprint(mol, fpSize=dimensions).ToBitString()]  # the bits converted to a list of numerical values


def ECFP(SMILES, mol, chirality, dimensions, radius=1):
    """
    Generates the Extended Connectivity Fingerprint (ECFP) for a molecule,
        optionally considering the ability to capture chirality, with a specified dimensions and radius
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param chirality: the ability to capture chirality
    :param dimensions:the length of the bit vector used to represent the molecule with ECFP
    :param radius: the radius of atom environments considered in the fingerprint
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith(
        "[H]/[N"))  # due to a bug of RDKit, a selection the stereoperception is needed
    return [int(bit) for bit in
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, useChirality=chirality,
                                                  nBits=dimensions).ToBitString()]  # the bits converted to a list of numerical values


def MHFP(SMILES, mol, encoder, dimensions, radius=1):
    """
    Generates the MinHash Fingerprint (MHFP) for a molecule, with its own encoder, a specified dimensions and radius
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param encoder: the encoder instance of the MHFP model
    :param dimensions:the length of the bit vector used to represent the molecule with MHFP
    :param radius: the radius of atom environments considered in the fingerprint
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith(
        "[H]/[N"))  # due to a bug of RDKit, a selection the stereoperception is needed
    try:
        fp = encoder.encode_mol(mol, radius=radius)
        return encoder.fold(fp, length=dimensions).tolist()  # the bits converted to a list of numerical values
    except:
        # if the mol cannot be handled by MHFP Encoder, it returns a list of 0.
        return [0] * 1024
