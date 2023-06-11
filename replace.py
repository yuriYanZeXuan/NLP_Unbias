import openai
import nltk

# Set the OpenAI API key
openai.api_key = ''

# Define a function to identify biased or emotionally charged words in a given sentence
def identify_biased_words(sentence):
    # Use the GPT-3 engine of OpenAI to generate text that lists biased or emotionally charged words in the sentence
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f"Identify biased or emotionally words in the following text: '{sentence}'",
      temperature=0.5,
      max_tokens=5
    )
    # Return a list of biased or emotionally charged words in the sentence
    return response.choices[0].text.strip().split(',')

# Define a function to generate a neutral word for a given biased word
def generate_neutral_word(biased_word):
    # Use the GPT-3 engine of OpenAI to generate a neutral word for the biased word
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f"Generate a neutral word for the word '{biased_word}'",
      temperature=0.5,
      max_tokens=5
    )
    # Return the generated neutral word
    return response.choices[0].text.strip()

# Define a function to replace biased words in a sentence with neutral words
def replace_biased_words_in_sentence(sentence, biased_to_neutral_dict):
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    new_words = []
    biased_words = identify_biased_words(sentence)
    for word in words:
        if word in biased_words:
            if word not in biased_to_neutral_dict:
                neutral_word = generate_neutral_word(word)
                biased_to_neutral_dict[word] = neutral_word
            new_words.append(biased_to_neutral_dict[word])
        else:
            new_words.append(word)
    # Return the sentence with biased words replaced with neutral words
    return ' '.join(new_words)

# Define a function to replace biased words in a text with neutral words
def replace_biased_words_in_text(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    new_sentences = []
    biased_to_neutral_dict = {}
    for sentence in sentences:
        new_sentence = replace_biased_words_in_sentence(sentence, biased_to_neutral_dict)
        new_sentences.append(new_sentence)
    # Return the text with biased words replaced with neutral words
    return ' '.join(new_sentences), biased_to_neutral_dict

# Read the original text from a file
with open(r'C:\Users\DELL\Desktop\NLP_Unbias-main\NLP_Unbias-main\rt-polaritydata\rt-polarity.neg.txt','r',encoding='Windows-1252') as f:
    original_text = f.read()

# Replace biased words in the text with neutral words
new_text, biased_to_neutral_dict = replace_biased_words_in_text(original_text)

# Write the new text with neutral words to a new file
with open(r'C:\Users\DELL\Desktop\NLP_Unbias-main\NLP_Unbias-main\rt-polaritydata\new_rt-polarity.neg.txt','w',encoding='Windows-1252') as f:
    f.write(new_text)

# Write the biased words and their corresponding neutral words to a txt file
with open('biased_to_neutral_dict.txt', 'w') as f:
    for biased_word, neutral_word in biased_to_neutral_dict.items():
        f.write(f'{biased_word}: {neutral_word}\n')
