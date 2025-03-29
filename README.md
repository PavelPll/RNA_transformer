# RNA_transformer
RNA Sequence Analysis Using Transformer Models
## Description
* Vanilla or/and BERT-like transformer implementation for RNA sequences analysis
* RNA embeddings arithmetic
* Peptidyl Transferase Center (PTC): structure and evolution 



## Getting Started

### Dependencies
* The starting point is [Vanilla Transformer implementation](https://github.com/hkproj/pytorch-transformer), used to translate phrases from English to Italian.
* [RNAcentral database](https://rnacentral.org)
* [RNA-FM: Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions](https://huggingface.co/multimolecule/rnafm)
* [ViennaRNA: predicting and comparing RNA secondary structures](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html)
* [StaVia (Via 2.0): single-cell trajectory inference method](https://pyvia.readthedocs.io/en/latest/pyVia-home.html)
* Windows 11, Visual Studio Code
* Torch

### Installing

* Install ViennaRNA from [here](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html)
* Install StaVia from [here](https://pyvia.readthedocs.io/en/latest/Installation.html)
```
git clone https://github.com/PavelPll/RNA_transformer.git
cd RNA_transformer
```
```
conda create -n rna python=3.10.16
conda activate rna
pip install -r requirements.txt
```

### Executing program

* #### Extract/Generate RNA sequence data 
    * Extraction from RNAcentral database
         ```
         cd scripts && python rna_data_extract.py
         ```
    * Generate real RNA sequences of interest:
        ```
        cd scripts && rna_data_extract_unique.py
        ```
    * Generate random/symmetric RNA sequences:
        ```
        cd scripts && python rna_data_extract_random.py
        cd scripts && rna_data_extract_symmetric.
        ```
    * Generate RNA hairpins:
        ```
        cd scripts && rna_data_extract_hairpin.py
        cd scripts && rna_data_extract_hairpin_tail.py
        ```
* #### Run RNA_transformer
    * Define transformer parameters in config models/RNA_transformer/config.py
      
        * Construct a composite loss
            ```
            "autoencoder_bert": False,
            "autoencoder_vanilla": True,
            "classification": True,
            ```
        * To use or not to use:
            * learnable loss weighting for the classification task
            * custom embeddings  
            * a mask token for training
        * etc
    * Run training and inference
        ```
        cd notebooks && RNA_transformer.ipynb
        ```
* #### Analysis
    * Calculate and plot RNA secondary structure:
        ```
        cd notebooks && RNA_plot.ipynb
        ```
    * Check embeddings arithmetic and the effect of basis:
        ```
        cd notebooks && embed_arithmetic_kdtree.ipynb
        cd notebooks && embed_arithmetic_basis.ipynb
        ```
    * Run Single-Cell Inspired Analysis of RNA
        ```
        cd notebooks && VIA_analysis.ipynb
        ```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

Pavel Polyakov  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)


eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee


# The project Sentiment Analysis (Containerized API)
Get/Extract the data from the internet **-->** Transform the data **-->** Load data into MongoDB database\
\
The file, **docker-compose.yml**, runs the container c1, with installed Python (Jupyter notebook), **two** MongoDB containers and **one** MYSQL container. The first container, c1, is used to run the code. The second container, my_mongo, is used to store the data (use my_mongo:27017 to access stored data from Python and Jupyter notebook). The third one, mongo-express, is used for MongoDB Graphic User Interface (http://localhost:8081). The fourth one is the container with MYSQL database. And the last one is the containner with flask API.
## Extraction
* The file src/Extract/**Extract_single.ipynb** (as well as its **.py** version) uses Selenium to extract the comments (author, score, ...) for certain categories (from categorie i to categorie j) of Software from this [website](https://www.capterra.fr/directory). The output see in data_csv/i_j.csv file. 
* The file src/Extract/**Extract_multiple.py** parallelizes the extraction process on multiple CPUs. It runs Extract_single.py in a parallel. Each CPU is taking care only of certain categories in order to accelerate data extraction.
## Transform
* The file src/Transform/**Transform.ipynb** concatenates all single extraction (see in data_csv/i_j.csv) to the final dataframe frame (data_csv/capterra.csv).
## Load
* The file src/Load/**MongoDB_load.py** loads capterra.csv (created in the previous Transform step) into MongoDB database and processes it.
* The file src/Load/**mysql_load.py** loads capterra.csv (created in the previous Transform step) into MYSQL database and processes it.
## Flask
* The file src/flask/**flask.ipynb** is used to run the requests (/status to get 1, /predict,...).
* The file requirements/flask/app/**main.py** contains the code to run ML prediction.

> [!NOTE]
> For technical side how to run the code please see the file ./How_to_run.txt.

