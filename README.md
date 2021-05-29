# NLP - Skill extraction

An school project whith ambition to extract skills from resumes.


## Dependancies

This project uses only Python and the required librairies are listed in **requirements.txt**

You can install them in your local environment by running `pip install -r requirements.txt`


# Data

For this project, I used data from [pôle-emploi.fr](https://www.pole-emploi.fr/) to build
a library of skills and expertises for french resumes.

The data mining process is explained in the first part of the notebook **playground.ipynb**.

The PDF files in the **CV_test** directory are samples from [Canva.com](https://www.canva.com/).


# Notebook

The notebook **playground.ipynb** keeps record of the different steps of the project and the differents
models tested. 

However, JupyterLab is not included in dependancies so you might want to install it.


# Application

A Streamlit application has been written to test the final results.

It can be run with `streamlit run app.py`

A little preview : 

![demo](app_demo.gif)