import streamlit as st
import re, contractions, nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

st.set_page_config(page_title="NLP Text Preprocessing", layout="wide")

st.title("📝 NLP Text Preprocessing Playground")

text = st.text_area("Enter Text", height=180)

col1, col2, col3 = st.columns(3)

with col1:
    lower = st.checkbox("Lowercase")
    punct = st.checkbox("Remove Punctuation")
    token = st.checkbox("Tokenization")

with col2:
    contract = st.checkbox("Expand Contractions")
    number = st.checkbox("Remove Numbers")
    stop = st.checkbox("Remove Stopwords")

with col3:
    space = st.checkbox("Remove Extra Spaces")
    lemma = st.checkbox("Lemmatization")
    stem = st.checkbox("Stemming")

output = text

if lower:
    output = output.lower()
if contract:
    output = contractions.fix(output)
if punct:
    output = re.sub(r"[^\w\s]", "", output)
if number:
    output = re.sub(r"\d+", "", output)
if space:
    output = re.sub(r"\s+", " ", output).strip()

if token or stop or lemma or stem:
    output = word_tokenize(output)

if stop:
    output = [i for i in output if i.lower() not in stop_words]
if lemma:
    output = [lemmatizer.lemmatize(i) for i in output]
if stem:
    output = [stemmer.stem(i) for i in output]

st.divider()

st.subheader("Processed Output")

if isinstance(output, list):
    st.code(output)
    st.write(" ".join(output))
else:
    st.write(output)

c1, c2 = st.columns(2)

c1.metric("Original Characters", len(text))
c2.metric(
    "Output Tokens" if isinstance(output, list) else "Output Characters",
    len(output)
)