# ==========================================
# Understanding LLMs
# Tiny Next-Word Prediction Model
# ==========================================


# --------------------------------------------------
# STEP 1: Training Data
# --------------------------------------------------

def get_training_data():
    # These sentences represent the data
    # our tiny model will learn from.
    sentences = [
        "python is powerful",
        "python is easy",
        "python is useful",
        "machine learning uses data",
        "machine learning needs data",
        "the sky is blue",
        "the sky is beautiful"
    ]

    return sentences


# --------------------------------------------------
# STEP 2: Tokenization
# --------------------------------------------------

def tokenize_sentence(sentence):
    # Split the sentence into individual words.
    #
    # Example:
    # "python is powerful"
    #
    # becomes:
    # ["python", "is", "powerful"]

    tokens = sentence.split()

    return tokens


# --------------------------------------------------
# STEP 3: Learn Word Relationships
# --------------------------------------------------

def build_model(sentences):
    # This dictionary will store the relationships
    # between words.
    #
    # Example:
    #
    # "python" -> ["is"]
    # "is"     -> ["powerful", "easy", "useful"]

    model = {}

    # Go through every training sentence
    for sentence in sentences:

        # Convert the sentence into tokens/words
        tokens = tokenize_sentence(sentence)

        # Go through the words using their indexes
        #
        # We stop at len(tokens) - 1 because
        # we always need a "next word".
        for i in range(len(tokens) - 1):

            # Current word
            current_word = tokens[i]

            # Word that comes after the current word
            next_word = tokens[i + 1]

            # If the current word doesn't exist
            # in our model yet, create an empty list.
            if current_word not in model:
                model[current_word] = []

            # Store the next word as a possible prediction
            model[current_word].append(next_word)

    return model


# --------------------------------------------------
# STEP 4: Predict the Next Word
# --------------------------------------------------

def predict_next_word(model, word):
    # Check whether we have seen this word
    # during training.
    if word not in model:

        # We don't know what comes after this word.
        return None

    # Get all possible words that appeared
    # after this word during training.
    possible_words = model[word]

    # For this simple example, just return
    # the first word we learned.
    #
    # Later we can make this smarter by
    # calculating probabilities.
    return possible_words[0]


# --------------------------------------------------
# STEP 5: Generate Text
# --------------------------------------------------

def generate_text(model, start_word, number_of_words):
    # Start the generated sentence with
    # the word provided by the user.
    generated_words = [start_word]

    # The current word is initially
    # the starting word.
    current_word = start_word

    # Generate one word at a time.
    for _ in range(number_of_words):

        # Ask our model to predict
        # what comes after the current word.
        next_word = predict_next_word(model, current_word)

        # If the model doesn't know the word,
        # stop generating.
        if next_word is None:
            break

        # Add the predicted word
        # to our generated sentence.
        generated_words.append(next_word)

        # The predicted word becomes
        # the new current word.
        current_word = next_word

    # Join all words together to create
    # one complete sentence.
    return " ".join(generated_words)


# --------------------------------------------------
# STEP 6: Run the Program
# --------------------------------------------------

if __name__ == "__main__":

    # 1. Get our training data
    sentences = get_training_data()

    print("Training Data:")
    print(sentences)

    # 2. Build our tiny language model
    model = build_model(sentences)

    print("\nLearned Model:")
    print(model)

    # 3. Test next-word prediction
    print("\nNext Word Prediction:")

    prediction = predict_next_word(model, "python")

    print("python ->", prediction)

    # 4. Generate a complete sentence
    print("\nGenerated Text:")

    result = generate_text(
        model,
        "python",
        2
    )

    print(result)