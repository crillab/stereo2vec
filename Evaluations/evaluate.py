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
from mhfp.encoder import MHFPEncoder
import numpy as np
import deepchem as dc
from mordred import Calculator, descriptors
import pandas as pd
from gensim.models import FastText
import argparse
import sys
import os
import random


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
if module_path not in sys.path:
    sys.path.append(module_path)

from Stereo2vec.stereo2vec_fingerprint import get_stereo2vec_vectors
from Fingerprints.learning_based_fingerprints import GCF, Mol2vec
from Fingerprints.procedural_fingerprints import MHFP, ECFP, MACCSFP, RDKitFP
from Fingerprints.molecular_descriptors import RDKitDescriptors, MordredDescriptors

# A number of line limitation to handle memory usage
CNTLIMIT = 1000000


def read_selected_lines(file_path, lines_selected):
    """
    Reads SMILES from a file that contains multiple SMILES
    :param file_path: the path of the file that contains multiple SMILES
    :param lines_selected: the lines that will be read from the file
    """
    with open(file_path, "r") as file:
        return [line.strip() for i, line in enumerate(file) if i in lines_selected]


def prepare_molecule(SMILES):
    """
    Prepares mol objects and kekulized SMILES
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith("[H]/[N"))
    molObject = Chem.MolFromSmiles(SMILES)
    if molObject is None:
        return None, None
    kekulized_SMILES = Chem.MolToSmiles(molObject, kekuleSmiles=True)
    molObject = Chem.MolFromSmiles(kekulized_SMILES)
    return kekulized_SMILES, molObject


def get_model(method_name, gensim_method, dimensions):
    """
    Loads the Gensim model of specific approach
    :param base_path: the location that models are saved
    :param method: the selected fingerprint method
    :param gensim_method: the selected gensim approach
    :param dimensions: the length of the bit vector used to represent the molecule
    """
    base_path = "../Models/"
    gm = "FT" if gensim_method == "FastText" else "W2V"

    if method_name == "IsoOrder2vec":
        model_path = base_path + "IsoSymbol2vec/" + gensim_method + "/IsoSymbol2vec_" + str(
            dimensions) + "_" + gm + ".model"
    else:
        model_path = base_path + method_name + "/" + gensim_method + "/" + method_name + "_" + str(
            dimensions) + "_" + gm + ".model"
    if gensim_method == "FastText":
        model = FastText.load(model_path)
    elif gensim_method == "Word2vec":
        model = np.load(model_path, allow_pickle=True)
    return model


def calculate_mordred(mol_list, calculator, large_set=False, batch_size=5000):
    """
    Calculates Mordred descriptors for a list of molecules
    :param mol_list: the list of RDKit mol objects
    :param calculator: the instance to calculate Mordred Descriptors
    :param large_set: to have a smaller molecules set for Mordred Descriptors to handle memory usage
    :param batch_size: the number of molecules in each batch for Mordred Descriptors
    """
    descriptors_list = []
    if not large_set:
        all_descriptors = MordredDescriptors(mol_list, calculator)
    else:
        for i in range(0, len(mol_list), batch_size):
            mol_batch = mol_list[i:i + batch_size]
            descriptor_df = MordredDescriptors(mol_batch, calculator)
            descriptors_list.append(descriptor_df)
        all_descriptors = pd.concat(descriptors_list, ignore_index=True)
    unique_descriptors = all_descriptors.drop_duplicates(keep=False)
    num_unique_descriptors = len(unique_descriptors)
    print(f"Processed molecules: {len(mol_list)}, Unique descriptor vectors: {num_unique_descriptors}")
    return unique_descriptors.values.tolist()


def compare_vectors(smiles_list, mol_list, method_name, large_set=False, batch_size=5000,
                    gensim_method="FastText", dimensions=1024,
                    radius=1):
    """
    Generates the vectors of selected fingerprint methods and compare for a list of molecules
    :param smiles_list: the list of SMILES representations of molecules
    :param mol_list: the list of RDKit mol objects
    :param method_name: the selected fingerprint method
    :param large_set: to have a smaller molecules set for Mordred Descriptors to handle memory usage, only works for Mordred.
    :param batch_size: the number of molecules in each batch for Mordred Descriptors, only works for Mordred
    :param gensim_method: the selected gensim approach, for Mol2vec and Stereo2vec approaches
    :param dimensions: the length of the bit vector used to represent the molecule
    :param radius: the radius of atom environments considered in the fingerprint
    """

    method_mapping = {
        # procedural fingerprints
        "MACCS": lambda SMILES, mol: MACCSFP(SMILES, mol),
        "RDKitFP": lambda SMILES, mol: RDKitFP(SMILES, mol, dimensions),
        "ECFP": lambda SMILES, mol: ECFP(SMILES, mol, False, dimensions, radius=radius),
        "ECFPChirality": lambda SMILES, mol: ECFP(SMILES, mol, True, dimensions, radius=radius),
        "MHFP": lambda SMILES, mol: MHFP(SMILES, mol, encoder, dimensions, radius=radius),

        # molecular descriptors
        "RDKitDescriptors": lambda SMILES, mol: RDKitDescriptors(SMILES, mol),
        "MordredDescriptors": lambda SMILES, mol: MordredDescriptors(mol, calculator),

        # learning based fingerprints
        "GCF": lambda SMILES, mol: GCF(SMILES, mol, featurizer, modelGCF),
        "Mol2vec": lambda SMILES, mol: Mol2vec(SMILES, mol, model, gensim_method, dimensions),
    }
    counts = {}
    if method_name == "MordredDescriptors":
        # it uses dataframe, the implementation needs a dataset
        return calculate_mordred(mol_list, calculator, large_set, batch_size)
    elif method_name in ["MolChiral2vec", "MolStereo2vec", "IsoString2vec", "IsoSymbol2vec",
                         "IsoOrder2vec"] or method_name in method_mapping:
        # all other methods can handle molecules one by one
        for SMILES, mol in zip(smiles_list, mol_list):
            if method_name in ["MolChiral2vec", "MolStereo2vec", "IsoString2vec", "IsoSymbol2vec", "IsoOrder2vec"]:
                vector = get_stereo2vec_vectors(SMILES, mol, method_name, gensim_method, dimensions, model)
            else:
                vector = method_mapping[method_name](SMILES, mol)
            vector_tuple = tuple(vector)
            if vector_tuple in counts:
                counts[vector_tuple]["count"] += 1
            else:
                counts[vector_tuple] = {"count": 1, "smiles": []}
            counts[vector_tuple]["smiles"].append(SMILES)
    else:
        print(method_name + " not in recognized")
        return
    not_duplicated = 0
    duplicatedSMILES = 0
    for sub_list_tuple, count_and_smiles in counts.items():
        # if the associated SMILES strings are different,
        # it means for these SMILES the generated vectors are same.
        # and it means the model's is inefficient.
        if count_and_smiles["count"] > 1:
            if len(set(count_and_smiles["smiles"])) == 1:
                duplicatedSMILES += count_and_smiles["count"]
        elif count_and_smiles["count"] == 1:
            not_duplicated += 1
    print(str(len(mol_list) - not_duplicated - duplicatedSMILES) + " out of " + str(len(mol_list)) + " different SMILES have the same numerical representations" )


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--file", type=str, default="../examples/smiles_examples.txt")
    CLI.add_argument("--method_name", type=str, required=True)
    CLI.add_argument("--gensim_method", type=str, default="FastText")
    CLI.add_argument("--dimensions", type=int, default=300)
    CLI.add_argument("--radius", type=int, default=1)
    CLI.add_argument("--large_set", action="store_true", default=False)
    CLI.add_argument("--batch_size", type=int, default=5000)
    args = CLI.parse_args()
    random.seed(42)
    file = args.file
    num_lines = sum(1 for _ in open(file))
    lines_selected = random.sample(range(num_lines), min(CNTLIMIT, num_lines))  # selecting random lines from the file
    smiles_list = read_selected_lines(file, set(lines_selected))  # reading SMILES from the file

    prepared = [prepare_molecule(smiles) for smiles in smiles_list]  # generating kekulized SMILES and mol objects.
    kekulized_SMILES, mol_list = zip(*prepared)
    kekulized_SMILES = list(kekulized_SMILES)
    mol_list = list(mol_list)

    if args.method_name in ["Mol2vec", "MolChiral2vec", "MolStereo2vec",
                            "IsoString2vec", "IsoSymbol2vec", "IsoOrder2vec"]:
        model = get_model(args.method_name, args.gensim_method, args.dimensions)
    elif args.method_name == "GCF":
        featurizer = dc.feat.ConvMolFeaturizer()
        features = featurizer.featurize(kekulized_SMILES)
        modelGCF = dc.models.GraphConvModel(n_tasks=1, mode="regression")
        # It can be replaced with a specificly generated model.
    elif args.method_name == "MHFP":
        encoder = MHFPEncoder()
    elif args.method_name == "MordredDescriptors":
        calculator = Calculator(descriptors)
        # It can be replaced with a selected subset of descriptors
    compare_vectors(kekulized_SMILES, mol_list, method_name=args.method_name, large_set=args.large_set,
                    batch_size=args.batch_size,
                    gensim_method=args.gensim_method,
                    dimensions=args.dimensions, radius=args.radius)
