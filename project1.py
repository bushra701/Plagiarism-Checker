import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def process(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def calculate(doc1, doc2):
    processed = [process(doc1), process(doc2)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]) * 100
    return similarity_score[0][0]  

#-----------------open files you want to compare as "1.txt" and "2.xt"--------------------------------
with open("1.txt") as file1, open("2.txt") as file2:
    doc1 = file1.read()
    doc2 = file2.read()

similarity = calculate(doc1, doc2)

print(f"Similarity: {similarity:.2f}%")