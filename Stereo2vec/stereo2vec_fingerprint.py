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
import numpy as np
import sys
import os

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
if module_path not in sys.path:
    print(module_path)
    sys.path.append(module_path)

from Stereo2vec.stereo2vec_parsers import MolStereo2vec, IsoSymbol2vec, MolChiral2vec


def get_stereo2vec_identifiers(SMILES, mol, selected_model):
    """
    Generates the identifiers for the selected Stereo2vec model  for a molecule with a specified dimensions
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param selected_model: the selected Stereo2vec model
    """
    Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith("[H]/[N"))
    if selected_model == "MolChiral2vec":
        molIdentifier = MolChiral2vec(mol, 1)
    elif selected_model == "MolStereo2vec":
        molIdentifier = MolStereo2vec(mol, 1)
    elif selected_model == "IsoString2vec":
        molIdentifier = Chem.MolToSmiles(mol, kekuleSmiles=True)
    elif selected_model == "IsoSymbol2vec" or selected_model == "IsoOrder2vec":
        molIdentifier = IsoSymbol2vec(SMILES)
    return molIdentifier


def get_stereo2vec_vectors(SMILES, mol, selected_model, gensim_method, dimensions, model):
    """
    Generates the final vectors for the selected Stereo2vec model with its special gensim method and a specified dimensions
    :param SMILES: the SMILES representation of the molecule, needed for the selection the StereoPerception
    :param mol: the RDKit mol object of the molecule
    :param selected_model: the selected Stereo2vec model
    :param gensim_method: selected gensim method, Word2vec or FastText
    :param dimensions: the number of dimensions used to represent the molecule
    """
    identifiers = get_stereo2vec_identifiers(SMILES, mol, selected_model)
    total_vector = np.zeros((dimensions,), dtype=np.float32)

    if selected_model == "MolStereo2vec":
        # MolStereo2vec returns a tuple of identifiers:
        # identifier of Morgan algorithm and the stereochemical information ('None' if there is no stereochemical information)
        # if there is no stereochemistry, we use simply the identifier of Morgan Algorithm
        # otherwise the identifier becomes identifier + _ + stereochemical information
        if gensim_method == "FastText":
            # FastText can handle Out-of-vocabulary words, no need to handle them.
            for identifier in identifiers:
                word_string = str(identifier[0]) # word selected as the first part of the tuple
                if identifier[1] is not None: # if there is a stereochemical information
                    word_string += "_" + identifier[1] # we append it to the identifier
                total_vector += model.wv.get_vector(word_string) # and we calculate the total vector
        elif gensim_method == "Word2vec":
            # Word2vec cannot handle Out-of-vocabulary words, therefore words must be searched in the model's vocablulary

            for identifier in identifiers:
                word_string = str(identifier[0]) # word selected as the first part of the tuple
                if identifier[1] is not None: # if there is a stereochemical information
                    word_string += "_" + identifier[1] # we append it to the identifier
                if word_string in model.wv.key_to_index: # if the final word is in vocabulary of the model
                    total_vector += model.wv.get_vector(word_string)  # we can calculate the total vector
                else:
                    # if the final word is in vocabulary of the model
                    try:
                        total_vector += model.wv.get_vector("UNSEEN") # the vector of "UNSEEN" is used when the vocabulary has it
                    except:

                        print("The identifier were not in the training corpus. "
                              "The training set does not contain any 'UNSEEN' term. "
                              "Therefore, the model cannot create any vector for the 'UNSEEN' term, "
                              "a 0*" + str(dimensions) + "vector is added to the final vector. "
                                                         "There will be no output for further 'UNSEEN' terms.")

    elif selected_model == "IsoString2vec":
        # IsoString2vec model uses a SMILES as a word, no need to generate identifiers
        return model.wv.get_vector(identifiers)

    elif selected_model == "IsoOrder2vec":
        all_vecs = []  # all vectors will be in a list to calculate slope
        if gensim_method == "FastText":
            # FastText can handle Out-of-vocabulary words, no need to handle them.
            for idx, identifier in enumerate(identifiers):
                vec = model.wv.get_vector(identifier)
                total_vector += vec
                all_vecs.append(vec)

        elif gensim_method == "Word2vec":
            # Word2vec cannot handle Out-of-vocabulary words, therefore words must be searched in the model's vocablulary
            for idx, identifier in enumerate(identifiers):
                if identifier in model.wv.key_to_index:
                    vec = model.wv.get_vector(identifier)
                    total_vector += vec
                    all_vecs.append(vec)
                else:
                    # if a word is not in vocabulary, it tries to replace them with "UNSEEN"
                    try:
                        vec = model.wv.get_vector("UNSEEN")
                        total_vector += vec
                        all_vecs.append(vec)
                    except:
                        # if the model does not contain any "UNSEEN", we will add a n-dimensional zero array is used for the final vector's calculation
                        print("The " + identifier + " were not in the training corpus. "
                                                    "The training set does not contain any 'UNSEEN' term. "
                                                    "Therefore, the model cannot create any vector for the 'UNSEEN' term, "
                                                    "a 0*" + str(
                            dimensions) + "vector is added to the final vector. "
                                          "There will be no output for further 'UNSEEN' terms.")
                        vec = np.zeros((dimensions,), dtype=np.float32)
                        total_vector += vec
                        all_vecs.append(vec)

        all_vecs_np = np.array(all_vecs)
        # the slope calculation
        x = np.arange(1, len(identifiers) + 1)[:, np.newaxis]
        y = all_vecs_np
        mean_x, mean_y = np.mean(x), np.mean(y, axis=0)
        cov_xy = np.mean(x * y, axis=0) - mean_x * mean_y
        var_x = np.mean(x ** 2) - mean_x ** 2
        slopes = cov_xy / var_x

        result_vector = np.zeros((2 * dimensions,), dtype=np.float32)
        result_vector[::2] = total_vector
        result_vector[1::2] = slopes
        # it returns a 2*n-dimensional list as [dimension 0, slope 0, dimension 1, slope 1, ... ]
        return result_vector

    else:
        # MolChiral2vec and IsoSymbol2vec models handle identifiers as theirself.
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
                    # if a word is not in vocabulary, it tries to replace them with "UNSEEN"
                    try:
                        total_vector += model.wv.get_vector("UNSEEN")
                    except:
                        # if the model does not contain any "UNSEEN", we will add a n-dimensional zero array is used for the final vector's calculation
                        print("The " + identifier + " were not in the training corpus. "
                                                    "The training set does not contain any 'UNSEEN' term. "
                                                    "Therefore, the model cannot create any vector for the 'UNSEEN' term, "
                                                    "a 0*" + str(
                            dimensions) + "vector is added to the final vector. "
                                          "There will be no output for further 'UNSEEN' terms.")
                        # total_vector += np.zeros((dimensions,), dtype=np.float32)
    return total_vector.tolist()
