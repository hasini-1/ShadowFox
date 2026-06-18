from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.util import bigrams
from collections import defaultdict

app = Flask(__name__)

# Load Dataset
with open("final.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

texts = [line.strip() for line in texts if line.strip()]

tokens = []

for text in texts:

    words_in_sentence = [
        word.lower()
        for word in word_tokenize(text)
        if word.isalpha()
    ]

    if len(words_in_sentence) > 1:
        tokens.append(words_in_sentence)

words = []

for sentence in tokens:
    words.extend(sentence)

# SpellChecker
spell = SpellChecker()
spell.word_frequency.load_words(words)

# Bigram Model
bigram_model = defaultdict(lambda: defaultdict(int))

for sentence in tokens:
    for w1, w2 in bigrams(sentence):
        bigram_model[w1][w2] += 1


def autocorrect_text(text):

    corrected_words = []

    for word in text.split():

        if word.lower() in spell:
            corrected_words.append(word)

        else:
            corrected_words.append(
                spell.correction(word)
            )

    return corrected_words


def predict_next_words(word, top_n=3):

    next_words = bigram_model[word]

    if not next_words:
        return []

    total = sum(next_words.values())

    sorted_words = sorted(
        next_words.items(),
        key=lambda x: x[1],
        reverse=True
    )

    result = []

    for w, freq in sorted_words[:top_n]:

        confidence = round(
            (freq / total) * 100,
            2
        )

        result.append(
            (w, confidence)
        )

    return result


@app.route("/", methods=["GET", "POST"])
def home():

    corrected_text = ""
    suggestions = []

    if request.method == "POST":

        user_input = request.form["text"]

        corrected_words = autocorrect_text(
            user_input
        )

        corrected_text = " ".join(
            corrected_words
        )

        last_word = corrected_words[-1].lower()

        suggestions = predict_next_words(
            last_word
        )

    return render_template(
        "index.html",
        corrected_text=corrected_text,
        suggestions=suggestions
    )


if __name__ == "__main__":
    app.run(debug=True)