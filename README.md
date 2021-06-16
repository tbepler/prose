# Protein Sequence Embeddings (ProSE)
Multi-task and masked language model-based protein sequence embedding models.

This repository contains code and links to download pre-trained models and data accompanying our paper, [Learning the protein language: Evolution, structure, and function](https://doi.org/10.1016/j.cels.2021.05.017). This extends from previous work, [Learning protein sequence embeddings using information from structure](https://openreview.net/pdf?id=SygLehCqtm).

## At a glance

Train bidirectional language model using the masked LM objective:
```
python train_prose_masked.py
```

Train bidirectional language model using the masked LM objective _and_ structure tasks:
```
python train_prose_multitask.py
```

Embed sequences using the pre-trained models:
```
python embed_sequences.py
```

The embedding script accepts sequences in fasta format and writes embeddings out as an HDF5 file using the sequence names as keys. Each sequence will have one dataset in the HDF5. Optionally, embeddings can be aggregated over the sequence positions to generate a fixed sized embedding for each sequence using the --pool argument.

For example, to embed the demo sequences in data/demo.fa to a file named data/demo.h5 using average pooling over each sequence (first, follow the instructions below to download the pre-trained models and install the python dependencies):
```
python embed_sequences.py --pool avg -o data/demo.h5 data/demo.fa
```

Note: your resulting demo.h5 may not match the provided demo.h5 exactly due to differences in rounding and non-determinism on different hardware, but your results should be close.

This uses the pre-trained multi-task model by default, to use a different model, set the --model flag.

Use the --help flag to get complete usage information.


## Setup instructions

### Download the pre-trained embedding models

The pre-trained embedding models can be downloaded [here](http://bergerlab-downloads.csail.mit.edu/prose/saved_models.zip).

They should be unzipped in the project base directory. By default, prose looks for the pre-trained models in the saved_models/ directory.

### Setup python environment

This code requires Python 3. I prefer Anaconda for ease of use. If you don't have conda installed already, get it [here](https://docs.conda.io/en/latest/miniconda.html).

1. (Optional but recommended) Make an anaconda environment for this workshop and activate it:
```
conda create -n prose python=3
source activate prose
```

2. Install the dependencies
```
conda env update --file environment.yml
```
or with pip
```
pip install -r requirements.txt
```

See the pytorch install [documentation](https://pytorch.org/get-started/locally/) for information on installing pytorch for different CUDA versions.

## Datasets

The training datasets are available at the links below.
- [SCOP data](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/scope.tar.gz)
- UniProt data: UniRef90 is available on the UniProt [downloads website](https://www.uniprot.org/downloads)

## Author
Tristan Bepler (<tbepler@gmail.com>)

## References

Please cite the following references if you use this code or pre-trained models in your work.

Bepler, T., Berger, B. Learning the protein language: evolution, structure, and function. Cell Systems 12, 6 (2021). https://doi.org/10.1016/j.cels.2021.05.017

<details><summary>Bibtex</summary><p>

```
@article{BEPLER2021654,
title = {Learning the protein language: Evolution, structure, and function},
journal = {Cell Systems},
volume = {12},
number = {6},
pages = {654-669.e3},
year = {2021},
issn = {2405-4712},
doi = {https://doi.org/10.1016/j.cels.2021.05.017},
url = {https://www.sciencedirect.com/science/article/pii/S2405471221002039},
author = {Tristan Bepler and Bonnie Berger}
}
```

</p></details>


Bepler, T., Berger, B. Learning protein sequence embeddings using information from structure. International Conference on Learning Representations (2019). https://openreview.net/pdf?id=SygLehCqtm


<details><summary>Bibtex</summary><p>

```
@inproceedings{
bepler2018learning,
title={Learning protein sequence embeddings using information from structure},
author={Tristan Bepler and Bonnie Berger},
booktitle={International Conference on Learning Representations},
year={2019},
}
```

</p></details>


## License

The source code and trained models are provided free for non-commercial use under the terms of the CC BY-NC 4.0 license. See [LICENSE](LICENSE) file and/or https://creativecommons.org/licenses/by-nc/4.0/legalcode for more information.


## Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact me at tbepler@gmail.com.
