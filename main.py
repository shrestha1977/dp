import io
import math
import pickle
import random
import re
import time
import wave
from collections import Counter

import librosa
import liwc
import numpy as np
import pandas as pd
import spacy
import speech_recognition as sr
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def generate_checklist(immediate_desc):
    doc = nlp(immediate_desc)
    items = set()
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADP"]:  # objects, actions, relations
            items.add(token.lemma_)
    return items


def calculate_retention_score(immediate_desc, delayed_desc):
    checklist = generate_checklist(immediate_desc)
    immediate_words = set(preprocess(immediate_desc))
    delayed_words = set(preprocess(delayed_desc))

    correct_initial = immediate_words & checklist
    correct_delayed = delayed_words & checklist
    retained = correct_initial & correct_delayed

    score = (len(retained) / len(correct_initial)) * 100 if correct_initial else 0
    return round(score, 2)


# === Helper functions ===
def load_images():
    return ["images/image1.jpg", "images/image2.jpg", "images/image3.jpg", "images/image4.jpg"]


def get_random_image():
    return random.choice(load_images())


# Tokenize function (same as before)
def tokenize(speech_text):
    for match in re.finditer(r'\w+', speech_text, re.UNICODE):
        yield match.group(0)


def classify_dementia_scale(cosine_similarity, dementia_prob):
    # First, apply the cosine_to_probability_piecewise transformation to map cosine similarity to a range
    def cosine_to_probability_piecewise(cosine_similarity):
        # Ensure cosine_similarity is a scalar
        if isinstance(cosine_similarity, (np.ndarray, list)):
            cosine_similarity = float(cosine_similarity[0])  # Convert to scalar if array-like

        if cosine_similarity <= 0.2:
            # Linear transformation for cosine similarity between 0 and 0.2
            probability = 90 + 25 * cosine_similarity
        elif 0.2 < cosine_similarity <= 0.6:
            # Linear transformation for cosine similarity between 0.2 and 0.6
            probability = 95 - 112.5 * (cosine_similarity - 0.2)
        else:
            # Linear transformation for cosine similarity between 0.6 and 1
            probability = 6 + 5 * (1 - cosine_similarity)

        return round(probability)

    # Ensure inputs are scalars
    if isinstance(cosine_similarity, (np.ndarray, list)):
        cosine_similarity = float(cosine_similarity[0])
    if isinstance(dementia_prob, (np.ndarray, list)):
        dementia_prob = float(dementia_prob[0])

    # Get the transformation of cosine similarity to probability
    similarity_prob = cosine_to_probability_piecewise(cosine_similarity)

    # Scale the dementia probability (between 0 and 1) to the range [32, 96]
    scaled_dementia_prob = 32 + (dementia_prob * (96 - 32))

    # Now, blend the similarity-based probability and the dementia probability
    # Higher cosine similarity should push the result higher in the range (more towards 96%)
    if cosine_similarity >= 0.4:
        # If cosine similarity is high, give more weight to the dementia probability
        final_probability = (0.6 * scaled_dementia_prob) + (0.4 * similarity_prob)
    else:
        # If cosine similarity is low, give more weight to the similarity-based probability
        final_probability = (0.4 * scaled_dementia_prob) + (0.6 * similarity_prob)

    return round(final_probability, 2)


# Function to compute category frequencies normalized by total word count
def compute_liwc_categories(speech_text, category_names, parse):
    tokens = list(tokenize(speech_text.lower()))  # Tokenize and convert to lowercase
    total_tokens = len(tokens)  # Total number of tokens

    # Initialize all categories with zero count
    category_frequencies = {category: 0 for category in category_names}

    # Count categories in the text
    category_counts = Counter(category for token in tokens for category in parse(token))

    # Update category frequencies based on counts
    for category, count in category_counts.items():
        category_frequencies[category] = count / total_tokens if total_tokens > 0 else 0

    return category_frequencies


# Function to compute Brunet's Index
def brunets_index(text, a=0.165):
    words = re.findall(r'\b\w+\b', text.lower())
    N = len(words)
    V = len(set(words))
    print("words:", words)

    if V == 0 or N == 0:
        return 0

    W = N ** (V ** -a)
    return W


# Function to compute Honoré's Statistic
def honores_statistic(text):
    words = re.findall(r'\b\w+\b', text.lower())
    N = len(words)
    print("words:", words)
    if N == 0:
        return 0

    word_counts = Counter(words)
    V = len(word_counts)
    V1 = sum(1 for count in word_counts.values() if count == 1)

    if V == 0 or V1 == V:
        return 0

    R = 100 * (math.log(N) / (1 - (V1 / V)))
    return R


# Function to compute Standardized Entropy
def standardized_entropy(text):
    words = re.findall(r'\b\w+\b', text.lower())
    N = len(words)
    print("words:", words)

    if N == 0:
        return 0

    word_counts = Counter(words)
    V = len(word_counts)

    if V == 1:
        return 0

    entropy = 0
    for count in word_counts.values():
        p = count / N
        entropy -= p * math.log(p, 2)

    max_entropy = math.log(V, 2)
    standardized_entropy_value = entropy / max_entropy
    return standardized_entropy_value


# Function to compute Root Type-Token Ratio (RTTR)
def root_type_token_ratio(text):
    words = re.findall(r'\b\w+\b', text.lower())
    N = len(words)  # Total number of words (tokens)
    print("words:", words)

    if N == 0:
        return 0  # Avoid division by zero for empty text

    unique_words = set(words)  # Number of unique words (types)
    V = len(unique_words)

    RTTR = V / math.sqrt(N)
    return RTTR


def extract_468_features(audio_file, sr=22050, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=sr)
    window_length = int(0.023 * sr)  # 23ms window
    hop_length = int(0.010 * sr)  # 10ms step size

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=window_length, window='hamming')
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    features_per_frame = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    min_features = np.min(features_per_frame, axis=1)
    max_features = np.max(features_per_frame, axis=1)
    mean_features = np.mean(features_per_frame, axis=1)
    std_features = np.std(features_per_frame, axis=1)

    summary_features_per_turn = np.hstack([min_features, max_features, mean_features, std_features])
    summary_features_per_turn = summary_features_per_turn.reshape(1, -1)  # Shape (1, 156)

    max_across_turns = np.max(summary_features_per_turn, axis=0)
    mean_across_turns = np.mean(summary_features_per_turn, axis=0)
    std_across_turns = np.std(summary_features_per_turn, axis=0)

    final_468_features = np.concatenate([max_across_turns, mean_across_turns, std_across_turns])  # Shape (468,)
    return final_468_features.reshape(1, -1)  # Reshape for model input


def generate_math_questions(num=6):
    questions = []
    for _ in range(num):
        a, b = random.randint(1, 20), random.randint(1, 20)
        op = random.choice(['+', '-'])
        question = f"{a} {op} {b}"
        answer = eval(question)
        questions.append((question, answer))
    return questions


# === Initialize session state ===
if "task" not in st.session_state:
    st.session_state.task = 1
if "selected_image" not in st.session_state:
    st.session_state.selected_image = get_random_image()
if "math_questions" not in st.session_state:
    st.session_state.math_questions = generate_math_questions()
if "math_answers" not in st.session_state:
    st.session_state.math_answers = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "voice_audio" not in st.session_state:
    st.session_state.voice_audio = None
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "timer" not in st.session_state:
    st.session_state.timer = 0
if "answered" not in st.session_state:
    st.session_state.answered = False

# Load LIWC dictionary
parse, category_names = liwc.load_token_parser('LIWC2007_English.dic')

# Load the model from the pickle file
with open('linguistic.pkl', 'rb') as f:
    linguistic = pickle.load(f)

# Load the model from the pickle file
with open('acoustic.pkl', 'rb') as f:
    acoustic = pickle.load(f)

# === TASK 1 ===
if st.session_state.task == 1:
    st.title("Dementia Detection - Task 1")
    st.image(st.session_state.selected_image, caption="Describe this image with your voice")

    st.write("🎤 Record your voice description:")
    audio_file = st.audio_input("Record audio here:")

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        st.session_state.voice_audio = audio_file

    if st.button("Next"):
        if st.session_state.voice_audio is not None:
            st.session_state.task = 2
            st.session_state.timer = 0
            st.session_state.answered = False
            st.rerun()
        else:
            st.warning("Please record your voice before proceeding.")

# === TASK 2 ===
elif st.session_state.task == 2:
    st.title("Dementia Detection - Task 2")
    questions = st.session_state.math_questions
    idx = st.session_state.current_question_index

    if idx < len(questions):
        question, correct_answer = questions[idx]
        st.subheader(f"Question {idx + 1}: {question} = ?")

        # Show timer and progress
        st.write(f"⏳ {10 - st.session_state.timer} seconds left")
        st.progress(st.session_state.timer / 10)

        if st.session_state.timer < 10:
            st.text_input("Enter your answer:", key=f"answer_{idx}")
            time.sleep(1)
            st.session_state.timer += 1
            st.rerun()
        else:
            user_answer = st.session_state.get(f"answer_{idx}", "")
            try:
                if user_answer.strip() != "" and int(user_answer.strip()) == correct_answer:
                    st.session_state.correct_count += 1
                st.session_state.math_answers.append(user_answer.strip())
            except:
                st.session_state.math_answers.append("")

            st.session_state.current_question_index += 1
            st.session_state.timer = 0
            st.rerun()

    # Show "Next" button only once — after all questions
    else:
        st.success("✅ All 6 questions completed.")
        st.write(f"Correct answers: {st.session_state.correct_count} / 6")
        if st.button("Next Task"):
            st.session_state.task = 3
            st.rerun()

# === TASK 3 ===
elif st.session_state.task == 3:
    st.title("Dementia Detection - Task 3")
    st.subheader("Describe the same image from Task 1 (without seeing it):")

    text_input = st.text_area("Describe the image in text:", value=st.session_state.text_input)
    st.session_state.text_input = text_input

    if st.button("Submit"):
        st.session_state.task = 4
        st.rerun()

# === FINAL SCORES ===
elif st.session_state.task == 4:
    st.title("Dementia Detection - Results")

    # Initialize the recognizer
    recognizer = sr.Recognizer()
    text = ""

    if st.session_state.voice_audio:
        # Display the audio player
        st.audio(st.session_state.voice_audio, format="audio/wav")

        # Read the audio data from the UploadedFile object
        audio_bytes = st.session_state.voice_audio.read()  # Read the content as bytes

        # Use BytesIO to create a file-like object from the bytes data
        audio_file_like = io.BytesIO(audio_bytes)

        # Load the audio data from the BytesIO object
        with sr.AudioFile(audio_file_like) as source:
            # Adjust for ambient noise (optional)
            recognizer.adjust_for_ambient_noise(source)

            # Record the audio data
            audio = recognizer.record(source)

        # Recognize the speech in the audio data
        try:
            # Using Google's Web Speech API for recognition
            text = recognizer.recognize_google(audio)
            st.write("Transcription: ", text)
        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError as e:
            st.write(f"Could not request results from the service; {e}")

        # Convert raw audio bytes to numpy array (assuming 16-bit PCM)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Create an in-memory WAV file using the wave module
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(22050)  # Match sample rate from original function
            wav_file.writeframes(audio_np.tobytes())

        buffer.seek(0)  # Move to the beginning of the file

        features_468 = extract_468_features(buffer)

    input_row_format = pd.read_csv('input_row_format.csv')

    # Apply the LIWC category frequency computation to each transcript
    liwc_results1 = compute_liwc_categories(text, category_names, parse)
    liwc_results2 = compute_liwc_categories(st.session_state.text_input, category_names, parse)

    # Convert the results into a DataFrame with one column for each LIWC category
    liwc_df1 = pd.DataFrame([liwc_results1])
    liwc_df2 = pd.DataFrame([liwc_results2])

    # Ensure that liwc_df has the same columns as X_train
    liwc_df1 = liwc_df1.reindex(columns=input_row_format.columns, fill_value=0)
    liwc_df2 = liwc_df2.reindex(columns=input_row_format.columns, fill_value=0)

    liwc_df1['brunets_index'] = brunets_index(text)
    liwc_df1['honores_statistic'] = honores_statistic(text)
    liwc_df1['standardized_entropy'] = standardized_entropy(text)
    liwc_df1['RTTR'] = root_type_token_ratio(text)

    liwc_df2['brunets_index'] = brunets_index(st.session_state.text_input)
    liwc_df2['honores_statistic'] = honores_statistic(st.session_state.text_input)
    liwc_df2['standardized_entropy'] = standardized_entropy(st.session_state.text_input)
    liwc_df2['RTTR'] = root_type_token_ratio(st.session_state.text_input)

    # 1. Write the list only once so you don’t repeat yourself
    desired_cols = [
        'pronoun', 'ipron', 'social', 'verb', 'auxverb', 'present', 'bio', 'adverb', 'nonfl',
        'affect', 'posemo', 'ppron', 'they', 'future', 'discrep', 'article', 'inhib', 'time',
        'conj', 'you', 'assent', 'i', 'shehe', 'quant', 'past', 'body', 'percept', 'feel',
        'family', 'see', 'home', 'negemo', 'filler', 'sad', 'achieve', 'hear', 'we', 'work',
        'health', 'anx', 'anger', 'swear', 'money', 'death', 'relig', 'friend', 'sexual',
        'brunets_index', 'honores_statistic', 'standardized_entropy', 'RTTR'
    ]

    # 2-A.  If you *know* all those columns exist after your first reindex:
    liwc_df1 = liwc_df1[desired_cols].copy()
    liwc_df2 = liwc_df2[desired_cols].copy()

    features_468 = features_468[:, [198, 1, 0]]

    y_prob1 = linguistic.predict_proba(liwc_df1)
    y_prob2 = linguistic.predict_proba(liwc_df2)
    y_prob3 = acoustic.predict_proba(features_468)

    # The output of predict_proba() is an array with two columns:
    # Column 0: Probability of the class '0' (no dementia)
    # Column 1: Probability of the class '1' (dementia)

    # To get the probability of dementia (class 1):
    dementia_prob1 = y_prob1[0][1]  # This gives the probability of class 1 (dementia)
    dementia_prob2 = y_prob2[0][1]

    pic_desc = {
        "images/image1.jpg": "This is a picture featuring a chaotic kitchen scene. The man is busy cutting veggies while the girls are cooking something. The dustbin is smelly and overfilled with waste. The mop and bucket are lying on the floor with water spilled over. The cat is sitting in the middle. There are many items on the table. The water in the pots in the oven is boiling. The kitchen is in complete disarray.",
        "images/image2.jpg": "This is a picture of a typical organized kitchen. The pans are neatly hanging on the wall. There is a fridge, oven, and chimney. The sink is kept clean with no dishes to wash. There's a small vase that adds to the aesthetics of the kitchen. The floor is made of vitrified checkered tiles, which are shiny and spick-free. Such an organized and neat place makes people happy.",
        "images/image3.jpg": "This picture features a mom with her two kids, a girl and a boy. The mom is busy doing the dishes with the sink overflowing with water, while the children are up to some naughty behavior. It seems both are busy stealing cookies from the shelf behind their mom's back. The boy is about to fall as the stool on which he is standing seems to topple while his sister is giggling or laughing and demands more cookies from her brother.",
        "images/image4.jpg": "This is a lively playground scene. All people seem so happy and cheerful, especially the children. Some are enjoying the slide while others are on the swings. A girl seems to be busy sharing something with her friend sitting on the bench, while her friend seems uninterested and more focused on eating. Two children are skipping ropes. An elder seems to have come with his baby in a stroller. One person seems to walk his dog. The two children seem thirsty, as they are quenching their thirst by drinking from the tap. Some children are playing tag. The person sitting on the bench seems to be speaking on the phone. Overall, the atmosphere seems merry."
    }

    desc_text = pic_desc[st.session_state.selected_image]

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF vectors for the texts
    tfidf_matrix1 = vectorizer.fit_transform([text, desc_text])
    tfidf_matrix2 = vectorizer.fit_transform([st.session_state.text_input, desc_text])

    # Compute cosine similarity between the two vectors
    context_score1 = cosine_similarity(tfidf_matrix1[0:1], tfidf_matrix1[1:2])[0][0]
    context_score2 = cosine_similarity(tfidf_matrix2[0:1], tfidf_matrix2[1:2])[0][0]

    linguistic_score = (dementia_prob1 + dementia_prob2) * 100 / 2
    acoustic_score = y_prob3[0][1] * 100
    calculation_score = (st.session_state.correct_count / 6) * 100
    memory_score = calculate_retention_score(text, st.session_state.text_input)

    total_score = (linguistic_score + acoustic_score + (100 - calculation_score) + (100 - memory_score)) / 4

    context_score = (context_score1 + context_score2) * 100 / 2
    final_score = classify_dementia_scale(context_score / 100, total_score / 100)

    st.metric("📝 Linguistic Score", f"{linguistic_score:.2f}%")
    st.caption("A higher linguistic score increases the risk of dementia")

    st.metric("🎤 Acoustic Score", f"{acoustic_score:.2f}%")
    st.caption("A higher acoustic score increases the risk of dementia")

    st.metric("➗ Calculation Score", f"{calculation_score:.2f}%")
    st.caption("A higher calculation score decreases the risk of dementia")

    st.metric("🧠 Memory Score", f"{memory_score:.2f}%")
    st.caption("A higher memory score decreases the risk of dementia")

    st.metric("🔍 Context Score", f"{context_score:.2f}%")
    st.caption("A higher context score decreases the risk of dementia")

    st.success(f"Total Dementia Risk Score: {final_score:.2f}%")

    st.write("✅ Thank you for participating!")

    if st.button("Restart"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
