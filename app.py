import json
import base64
from re import split, sub
from pathlib import Path
from sys import maxunicode
from itertools import chain

from unicodedata import category
from unidecode import unidecode

import pandas as pd
import numpy as np

from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

import streamlit as st

import fitz

st.set_page_config(layout="wide")


# =========================================================================================================
# Corpus class 

class Corpus:
    
    __punctuation_table = {char: " " for char in range(maxunicode) if category(chr(char)).startswith("P")}
    __stopwords = set(Path("stopwords-fr.txt").read_text().split("\n"))
    __stemmer = FrenchStemmer(ignore_stopwords=True)
    
    def __init__(self, sentences: set, remove_accents: bool = True, stemming: bool = False):
        self.sentences = list(sentences)
        self.corpus = list()
        self.vocab = set()
        self.remove_accents = remove_accents
        self.stemming = stemming
        
    def __iter__(self):
        return (s for s in self.corpus)
    
    @property
    def tokenized_documents(self):
        return [s.split() for s in self.corpus]
        
    def clean_sentence(self, sentence: str):
        sentence = sentence.lower().translate(self.__punctuation_table)
        words = [word for word in sentence.split() if word not in self.__stopwords]
        if self.stemming:
            words = [self.__stemmer.stem(word) for word in words]
        if self.remove_accents:
            words = [unidecode(word) for word in words]
        return " ".join(words)
        
    def build_corpus(self):
        for i, sentence in enumerate(self.sentences):
            clean_sentence = self.clean_sentence(sentence)
            self.corpus.append(clean_sentence)
            self.vocab.update(clean_sentence.split())

            
# =========================================================================================================
# Load ROME sheets

@st.cache
def load_rome_sheets():
    path = Path('./data/')
    sheets = []
    for file in path.glob("?/?????.json"):
        with file.open() as j:
            sheets.append(json.load(j))
    return sheets


# =========================================================================================================
# Extract skills and knowledges 

@st.cache
def extract_skills_and_knowledges(sheets):
    skills = set(chain(
        *[sheet.get("competences", []) for sheet in sheets], 
        *[sheet.get("competences_extra", []) for sheet in sheets]))
    knwlgs = set(chain(
        *[sheet.get("expertises", []) for sheet in sheets], 
        *[sheet.get("expertises_extra", []) for sheet in sheets]))
    return skills, knwlgs


# =========================================================================================================
# Build models and comparison tools

@st.cache
def build_comparator(documents: set):
    
    corpus = Corpus(documents, stemming=True)
    corpus.build_corpus()
    
    tfidf = TfidfVectorizer(lowercase=False, vocabulary=corpus.vocab)
    w2v = Word2Vec(
        corpus.tokenized_documents,
        min_count=1,
        vector_size=50,
        epochs=100,
        workers=4,
    )
    tf_mat = tfidf.fit_transform(corpus)
    vecs = np.array([w2v.wv[word] for word in tfidf.get_feature_names()])
    embeddings = tf_mat @ vecs / tf_mat.sum(axis=1).reshape(-1, 1)
    
    def comparator(sentence: str) -> tuple[str, int]:
        idfs = tfidf.transform([corpus.clean_sentence(sentence)]).tocoo()
        if idfs.getnnz() == 0:
            return ("", np.inf)
        vec = np.sum([idf * vecs[index] for index, idf in zip(idfs.col, idfs.data)], axis=0) / idfs.sum()
        cosines = cosine_similarity(embeddings, vec.reshape(1, -1)).flatten()
        return max(zip(corpus.sentences, cosines), key=lambda s: s[1])
    
    return comparator


# =========================================================================================================
# Document extraction

def extract_sentences(file, filetype: str) -> list[str]:
    
    sentences = []
    
    if filetype == "TXT":
        sentences.extend(file.decode("utf-8").split("\n"))
    
    if filetype == "PDF":
        doc = fitz.open(stream=file, filetype="pdf")
        for block in doc.loadPage(0).get_text("blocks"):
            block_text = sub(r"<[^>]*>|\n", " ", block[4])
            for snt in split(r"[.!?-]\s", block_text):
                if cst := snt.strip():
                    sentences.append(cst)  
                    
    return sentences


# =========================================================================================================
# Initialization

# get source information
rome_sheets = load_rome_sheets()
skills, knwlgs = extract_skills_and_knowledges(rome_sheets)

# build comparators
skill_comparator = build_comparator(skills)
knwlg_comparator = build_comparator(knwlgs)


# =========================================================================================================
# Sidebar

st.sidebar.title("Importer un CV")

filetype = st.sidebar.selectbox(
    "Type de fichier",
    ["PDF", "TXT"]
)

uploaded_file = st.sidebar.file_uploader('Sélectionez votre CV', type=filetype)

analyze = st.sidebar.button("Lancer l'analyse")


# =========================================================================================================
# Main page

if analyze and uploaded_file is not None:
    
    
    file = uploaded_file.read()
    sentences = extract_sentences(file, filetype)

    resume_skills = []
    resume_knwlgs = []
    
    for snt in sentences:
        best_skill = skill_comparator(snt)
        best_knwlg = knwlg_comparator(snt)
        if best_skill[1] != np.inf:
            resume_skills.append(best_skill)
        if best_knwlg[1] != np.inf:
            resume_knwlgs.append(best_knwlg)
        
    top_skills = sorted(dict(sorted(resume_skills, key=lambda s: s[1])).items(), key=lambda s: s[1], reverse=True)[:5]
    top_knwlgs = sorted(dict(sorted(resume_knwlgs, key=lambda s: s[1])).items(), key=lambda s: s[1], reverse=True)[:5]
    
    col1, col2 = st.beta_columns((2, 1))
    
    if filetype == "PDF":
        base64_pdf = base64.b64encode(file).decode('utf-8')
        resume_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#toolbar=0" width="90%" height="700" type="application/pdf">' 
        col1.write(resume_display, unsafe_allow_html=True)
    if filetype == "TXT":
        col1.write(file.decode("utf-8"))
        
    col2.markdown("## Éléments extraits :")
    
    exp_sk = col2.beta_expander("Compétences")
    for sk, sc in top_skills:
        exp_sk.write(f"{sk} (cohérence {100*sc:.1f}%)")
        exp_sk.progress(sc)
        
    exp_kn = col2.beta_expander("Expertises")
    for kn, sc in top_knwlgs:
        exp_kn.write(f"{kn} (cohérence {100*sc:.1f}%)")
        exp_kn.progress(sc)
    
    
    