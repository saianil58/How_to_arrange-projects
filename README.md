# How to Arrange ML Projects:

Jupyter notebooks are great! but they wont help you if you want to run things fast or in parallel or in a LINUX env.

In Industry projects in general follow the below example.

├── input
│ ├── train.csv
│ └── test.csv
├── src
│ ├── create_folds.py
│ ├── train.py
│ ├── inference.py
│ ├── models.py
│ ├── config.py
│ └── model_dispatcher.py
├── models
│ ├── model_rf.bin
│ └── model_et.bin
├── notebooks
│ ├── exploration.ipynb
│ └── check_data.ipynb
├── README.md
└── LICENSE

Let’s see what these folders and file are about.

input/: This folder consists of all the input files and data for your machine learning project. If you are working on NLP projects, you can keep your embeddings here. If you are working on image projects, all images go to a subfolder inside this folder.

src/: We will keep all the python scripts associated with the project here. If I talk about a python script, i.e. any *.py file, it is stored in the src folder.

models/: This folder keeps all the trained models.

notebooks/: All jupyter notebooks (i.e. any *.ipynb file) are stored in the notebooks folder.

README.md: This is a markdown file where you can describe your project and write instructions on how to train the model or to serve this in a production environment.

LICENSE: This is a simple text file that consists of a license for the project, such as MIT, Apache, etc. 

This project shows an simple classification example of Predicting if a person has diabetes or not.
