import ssl
import nltk

# Disable SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download corpora required by TextBlob
nltk.download('brown')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('conll2000')
nltk.download('movie_reviews')

print("✅ TextBlob corpora downloaded successfully.")
