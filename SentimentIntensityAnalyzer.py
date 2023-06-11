import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Function to read the file content
def read_file(file_path):
    with open(file_path, 'r', encoding='Windows-1252') as f:
        return f.read()

# Reading your file
file_path = r'C:\Users\DELL\Desktop\NLP_Unbias-main\NLP_Unbias-main\rt-polaritydata\new_rt-polarity.neg.txt'  # replace with your file path
text = read_file(file_path)

sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text)

if sentiment['compound'] > 0: 
    print("Positive sentiment detected.")
elif sentiment['compound'] == 0: 
    print("Neutral sentiment detected.")
else: 
    print("Negative sentiment detected.")

print(sentiment)