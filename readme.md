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

This project depends on the following Python libraries:
- **RDKit** version 2023.03.1 - pip install rdkit==2023.03.1
- **NumPy** version 1.24.3 - pip install numpy==1.24.3
- **Pandas** version 1.3.4 - pip install pandas==1.3.4
- **DeepChem** version 2.7.1 - pip install deepchem==2.7.1
- **Tensorflow** version 2.13.1 - pip install tensorflow==2.13.1
- **Gensim** version 4.1.2 - pip install gensim==4.1.2
- **Mordred** version 1.2.0 - pip install mordred==1.2.0
- **MHFP** version 1.9.6 - pip install mhfp==1.9.6
- **Mol2vec** version 0.1 - pip install mol2vec==0.1
- **networkx** version 2.5 - pip install networkx==2.5
- **scikit-learn** version 1.3.0 - pip intsall scikit-learn==1.3.0
- **scipy** version 1.8.1 - pip intsall scikit-learn==1.8.1
- **h5py** version 3.10.0 - pip intsall scikit-learn==3.10.0


To train a model, the ./Evaluations/train.py script must be used.
It takes 7 arguments, 4 of them are required:
    method_name: the selected method, "Mol2vec", "MolChiral2vec", "MolStereo2vec", "IsoString2vec", "IsoSymbol2vec" or "IsoOrder2vec" (required)
    base_path: the path that models will be saved (optional, by default it's "./examples/") 
    file: the file that contains multiple SMILES (optional, by default it's "./examples/smiles_examples.txt", which has 1000 random SMILES)
    generate_corpus: the corpus generation option (required)
    gensim_method: the gensim method, "FastText" or "Word2vec" (required)
    train: the model training option (required)
    dimensions: the number of dimensions used to represent the molecule (optional, by default it's 300)

Example usage:
    python train.py --method_name "MolChiral2vec" --gensim_method "FastText" --train "yes" --generate_corpus "yes"; -> To train a 300-dimensional MolChiral2vec model with FastText approach, with its own corpus generation
    python train.py --method_name "MolStereo2vec" --gensim_method "Word2vec" --train "yes" --generate_corpus "no" --dimensions 100; -> To train a 100-dimensional MolStereo2vec with Word2vec, with an existing corpus

To evaluate a model ./Evaluations/evaluate.py script must be used: it generates the numerical vectors for the selected model and selected file that contains multiple smiles and then compares the number of duplicated data. It uses prebuilt models.
It takes 7 arguments, 1 of them are required:
    file: the file that contains multiple SMILES (optional, by default it's "./examples/smiles_examples.txt", which has 1000 random SMILES)
    method_name: the selected method (required)
        Molecular Descriptors: "RDKitDescriptors" or "MordredDescriptors";
        Procedural Fingerprints: "MACCS", "RDKitFP", "ECFP", "ECFPChirality" or "MHFP";
        Learning-based Fingerprints: "GCF" or "Mol2vec"; 
        Stereo2vec: "MolChiral2vec", "MolStereo2vec", "IsoString2vec", "IsoSymbol2vec" or "IsoOrder2vec"
    gensim_method: the gensim method, "FastText" or "Word2vec" (optional, by default it's FastText)
    dimensions: the number of dimensions used to represent the molecule (optional, by default it's 300)
    radius: the radius of atom environments considered in the fingerprint (optional, by default it's 1)
    large_set: to have a smaller molecules set for Mordred Descriptors to handle memory usage (optional, by default it's False, it will just work for Mordred Descriptors)
    batch_size: the number of molecules in each batch for Mordred Descriptors (optional, by default it's 5000, it will just work for Mordred Descriptors)

Example usage:
    python evaluate.py --method_name "MACCS";
    python evaluate.py --method_name "ECFPChirality" --dimensions 1024;
    python evaluate.py --method_name "ECFPChirality" --dimensions 1024 --radius 2;
    python evaluate.py --method_name "RDKitDescriptors";
    python evaluate.py --method_name "MordredDescriptors" --large_set --batch_size 10000;
    python evaluate.py --method_name "MolStereo2vec";
    python evaluate.py --method_name "MolStereo2vec" --dimensions 200;
    python evaluate.py --method_name "IsoSymbol2vec" --gensim_method "Word2vec" --dimensions 300; 