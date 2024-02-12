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

from mol2vec.features import mol2alt_sentence
import deepchem as dc
import numpy as np
from rdkit import Chem


def GCF(SMILES, mol, featurizer, model):
    """
    Generates the Graph Convolutional Fingerprint (GCF) for a molecule with a specified dimensions
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param featurizer: A featurizer that transforms input molecule into a graph-based representation
    :param model: the GCF model
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith("[H]/[N")) # due to a bug of RDKit, a selection the stereoperception is needed
    dataset = dc.data.NumpyDataset(X=featurizer.featurize([mol]))
    fingerprints = model.predict_embedding(dataset)
    return fingerprints[0].tolist()


def Mol2vec(SMILES, mol, model, gensim_method, dimensions):
    """
    Generates the final vectors of Mol2vec for a molecule with a specified dimensions
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param model: the Mol2vec model
    :param gensim_method: selected gensim method, Word2vec or FastText
    :param dimensions: the length of the bit vector used to represent the molecule with Mol2vec
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith("[H]/[N")) # due to a bug of RDKit, a selection the stereoperception is needed
    identifiers = mol2alt_sentence(mol, 1) # calculation the identifiers
    total_vector = np.zeros((dimensions,), dtype=np.float32)
    if gensim_method == "FastText":
        # FastText can handle Out-of-vocabulary words, no need to handle them.
        for identifier in identifiers:
            total_vector += model.wv.get_vector(identifier)
    elif gensim_method == "Word2vec":
        # Word2vec cannot handle Out-of-vocabulary words, therefore words must be searched in the model's vocablulary
        for identifier in identifiers:
            if identifier in model.wv.key_to_index:
                total_vector += model.wv.get_vector(identifier)
            else:
                # if a word is not in vocabulary, it tries to replace them with UNSEEN
                try:
                    total_vector += model.wv.get_vector("UNSEEN")
                except:
                    # if the model does not contain any UNSEEN, we will add a n-dimensional zero array is used for the final vector's calculation
                    print("The " + identifier + " were not in the training corpus. "
                                                "The training set does not contain any 'UNSEEN' term. "
                                                "Therefore, the model cannot create any vector for the 'UNSEEN' term, "
                                                "a 0*" + str(
                        dimensions) + "vector is added to the final vector. "
                                      "There will be no output for further 'UNSEEN' terms.")
                    total_vector += np.zeros((dimensions,), dtype=np.float32)

    return total_vector.tolist() # np.array transformed to a list
