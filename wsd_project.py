import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# DOWNLOAD NLTK DATA 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# STREAMLIT UI
st.set_page_config(page_title="WSD System", page_icon="🧠")

st.title("🧠 Word Sense Disambiguation System")
st.caption("Improved Semantic Lesk Algorithm + WordNet Explorer")

sentence = st.text_input("Enter Sentence")
word = st.text_input("Enter Ambiguous Word")

# POS CONVERSION 
def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


# BUILD SIGNATURE 
def get_signature(sense):

    signature = set()

    signature.update(word_tokenize(sense.definition().lower()))

    for ex in sense.examples():
        signature.update(word_tokenize(ex.lower()))

    for hyper in sense.hypernyms():
        signature.update(word_tokenize(hyper.definition().lower()))

    for hypo in sense.hyponyms():
        signature.update(word_tokenize(hypo.definition().lower()))

    signature = {
        lemmatizer.lemmatize(w)
        for w in signature
        if w not in stop_words
    }

    return signature


# CONTEXT WINDOW 
def get_context(sentence, word):

    tokens = word_tokenize(sentence.lower())

    try:
        index = tokens.index(word.lower())
    except:
        return set(tokens)

    window = tokens[max(0, index-4): index+5]

    context = {
        lemmatizer.lemmatize(w)
        for w in window
        if w not in stop_words
    }

    return context


# IMPROVED LESK 
def improved_lesk(sentence, word, pos=None):

    context = get_context(sentence, word)

    best_sense = None
    best_score = 0
    sense_scores = []

    financial_keywords = {"money", "cash", "deposit", "account", "loan"}

    for sense in wn.synsets(word, pos=pos):

        signature = get_signature(sense)

        overlap = context.intersection(signature)

        score = len(overlap)

        # semantic rule bonus
        if context.intersection(financial_keywords):
            if "financial" in sense.definition() or "money" in sense.definition():
                score += 3

        sense_scores.append((sense, score))

        if score > best_score:
            best_score = score
            best_sense = sense

    sense_scores.sort(key=lambda x: x[1], reverse=True)

    return best_sense, best_score, sense_scores[:3]


# GET ALL MEANINGS 
def get_all_meanings(word, pos=None):

    meanings = []

    for syn in wn.synsets(word, pos=pos):

        synonyms = [
            lemma.replace("_", " ")
            for lemma in syn.lemma_names()
        ]

        meanings.append({
            "definition": syn.definition(),
            "synonyms": list(set(synonyms))
        })

    return meanings


# MAIN BUTTON 
if st.button("Find Meaning"):

    if sentence and word:

        tokens = word_tokenize(sentence)
        tagged_words = pos_tag(tokens)

        wn_pos = wn.NOUN

        for w, t in tagged_words:
            if w.lower() == word.lower():
                wn_pos = get_wordnet_pos(t)

        lemma_word = lemmatizer.lemmatize(word.lower(), pos=wn_pos)

        sense, score, top_senses = improved_lesk(
            sentence, lemma_word, wn_pos
        )

        # BEST MEANING 
        if sense:

            st.success("✅ Correct Meaning")

            st.write("### Definition")
            st.write(sense.definition())

            st.write(f"**Confidence Score:** {score}")

            # SYNONYMS OF BEST SENSE 
            st.subheader("Synonyms (Selected Meaning)")

            synonyms = sense.lemma_names()

            cols = st.columns(3)

            for i, syn in enumerate(synonyms):
                cols[i % 3].markdown(
                    f"""
                    <div style="
                        background-color:#f0f2f6;
                        padding:10px;
                        border-radius:10px;
                        text-align:center;
                        margin:5px;
                        font-weight:bold;">
                        {syn.replace('_',' ')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # TOP COMPETING MEANINGS 
            st.subheader("Top Competing Meanings")

            for s, sc in top_senses:
                st.write(f"• {s.definition()} (Score: {sc})")

        # ALL POSSIBLE MEANINGS 
        st.subheader("All Possible Meanings & Synonyms")

        all_meanings = get_all_meanings(lemma_word, wn_pos)

        for meaning in all_meanings:

            if sense and meaning["definition"] == sense.definition():
                st.success("Selected Meaning")

            st.write(meaning["definition"])

            cols = st.columns(4)

            for i, syn in enumerate(meaning["synonyms"]):
                cols[i % 4].markdown(
                    f"""
                    <div style="
                        background-color:#eef2ff;
                        padding:8px;
                        border-radius:8px;
                        text-align:center;
                        margin:4px;">
                        {syn}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:
        st.warning("⚠️ Please enter both sentence and word")