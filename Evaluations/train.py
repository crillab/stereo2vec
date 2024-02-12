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


from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from mol2vec.features import mol2alt_sentence
import argparse
import time
import pickle
from collections import defaultdict
from rdkit import Chem
import sys
import os


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
if module_path not in sys.path:
    sys.path.append(module_path)

from Stereo2vec.stereo2vec_parsers import MolStereo2vec, MolChiral2vec, IsoSymbol2vec


def corpus_generation(file, method_name, save_file):
    """
    Generates the corpus for the selected model
    :param file: the file that has multiple SMILES
    :param method_name: the selected fingerprint method_name
    :param save_file: the file that will be used to save corpus
    """
    tagged_documents = []
    with open(file) as f:
        for line in f:
            # Chem.SetUseLegacyStereoPerception() should be false to capture all chiral center in RDKit version 2023.03.1.
            # However,  due to a bug, H atom attached to an N lose stereochemistry information.
            # Therefore,  Chem.SetUseLegacyStereoPerception(True) is used for this kind of molecules.
            SMILES = line.split("\n")[0]
            Chem.SetUseLegacyStereoPerception(SMILES.startswith("[H]/N") or SMILES.startswith("[H]/[N"))
            molObject = Chem.MolFromSmiles(SMILES)
            smileNormalised = Chem.MolToSmiles(molObject, kekuleSmiles=True)
            molObject = Chem.MolFromSmiles(smileNormalised)
            if method_name == "Mol2vec":
                tagged_documents.append(mol2alt_sentence(molObject, 1))
            elif method_name == "MolChiral2vec":
                tagged_documents.append(MolChiral2vec(molObject, 1))
            elif method_name == "MolStereo2vec":
                tagged_documents.append(MolStereo2vec(molObject, 1))
            elif method_name == "IsoString2vec":
                tagged_documents.append([smileNormalised])
            elif method_name == "IsoSymbol2vec":
                parsedSMILES = IsoSymbol2vec(smileNormalised)
                tagged_documents.append(parsedSMILES)
    with open(save_file, "wb") as fp:
        pickle.dump(tagged_documents, fp)
    return tagged_documents


def train_model(corpus, save_file, method_name, gensim_method="FastText",  dimensions=300):
    """
    Trains the model for selected method and selected gensim approach within a selected dimension 
    :param corpus: the file that has multiple SMILES
    :param save_file: the file that will be used to save corpus
    :param method_name: the selected fingerprint method_name
    :param gensim_method: the selected gensim approach, for Mol2vec and Stereo2vec approaches
    :param dimensions: the number of dimensions used to represent the molecule
    """

    model_class = FastText if gensim_method == "FastText" else (
        None if method_name == "IsoString2vec" and gensim_method == "Word2vec" else Word2Vec)

    if model_class == None:
        return

    if gensim_method == "Word2vec":
        # the rare word handling approach for Word2vec
        word_freq = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                word_freq[word] += 1
        new_corpus = []
        for sentence in corpus:
            new_sentence = []
            for word in sentence:
                if word_freq[word] < 3:
                    new_sentence.append("UNSEEN")
                else:
                    new_sentence.append(word)
            new_corpus.append(new_sentence)
        corpus = new_corpus
    if method_name == "MolStereo2vec":
        # For MolStereo2vec method_name, the tuples is changed with a single element which contains identifier and stereochemical information
        corpus_strings = []
        for doc in corpus:
            doc_strings = []
            for word_tuple in doc:
                if word_tuple != "UNSEEN":
                    word_string = str(word_tuple[0])
                    if word_tuple[1] is not None:
                        word_string += "_" + word_tuple[1]
                    doc_strings.append(word_string)
                else:
                    doc_strings.append("UNSEEN")
            corpus_strings.append(doc_strings)
        corpus = corpus_strings
    model = model_class(corpus, sg=1, window=10, vector_size=dimensions, min_count=1, workers=1)
    model.save(save_file)


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--method_name", type=str, required=True)
    CLI.add_argument("--base_path", type=str, default="../examples/")
    CLI.add_argument("--file", type=str, default="../examples/smiles_examples.txt")
    CLI.add_argument("--generate_corpus", type=str, default=[], required=True)
    CLI.add_argument("--gensim_method", type=str, default=[], required=True)
    CLI.add_argument("--train", type=str, default=[], required=True)
    CLI.add_argument("--dimensions", type=int, default=300)

    args = CLI.parse_args()

    start_time = time.time()
    file = args.file
    method_name = args.method_name
    gensim_method = args.gensim_method
    training = args.train
    generate_corpus = args.generate_corpus
    dimensions = args.dimensions
    base_path = args.base_path

    if not (method_name in ["Mol2vec", "MolChiral2vec", "MolStereo2vec",
                            "IsoString2vec", "IsoSymbol2vec", "IsoOrder2vec"]):
        print("method_name value is incorrect")
        exit()
    if not (gensim_method in ["FastText", "Word2vec"]):
        print("Only FastText or Word2vec can be used as gensim method_name.")
        exit()
    if not (generate_corpus in ["yes", "no"]):
        print("--generate_corpus must be 'yes' or 'no'.")
        exit()
    if not (training in ["yes", "no"]):
        print(training)
        print("--train must be 'yes' or 'no'.")
        exit()

    if method_name == "IsoOrder2vec":
        method_name = "IsoSymbol2vec"
        print("The model of Order Sensitive Token Based is same with Token Based. "
              "However, the final vectors will change. In training, there is no changement.")
    directory_path = base_path + method_name + "/" + gensim_method + "/"
    os.makedirs(directory_path, exist_ok=True)
    save_file = directory_path + file.split("/")[-1].split(".")[0] + ".corpus"

    if generate_corpus == "yes":
        print("Corpus generation is started.")
        tagged_documents = corpus_generation(file, method_name, save_file)
        print("Corpus has been generated succesfully.")
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time: " + str(round(total_time, 2)) + " seconds")
        start_time = time.time()

    if training == "yes":
        print("training")
        if generate_corpus == "no":
            # if the corpus not generated in this execution, read it from the saved file.
            with open(save_file, "rb") as fp:
                tagged_documents = pickle.load(fp)
        model_file = base_path + method_name + "/" + gensim_method + "/" + str(dimensions) + ".model"
        train_model(tagged_documents, model_file, method_name, gensim_method, dimensions)
        print("The model has been generated succesfully.")

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time: " + str(round(total_time, 2)) + " seconds")
