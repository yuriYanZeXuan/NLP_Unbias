from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Define a function to segment long text into paragraphs and extract their embeddings
def embed_text(text, model, tokenizer):
    tokens = tokenizer(text, truncation=True, padding='longest', max_length=512, return_tensors='pt')
    total_tokens = tokens["input_ids"].shape[1]
    num_segments = total_tokens // 512 + 1 if total_tokens % 512 else total_tokens // 512

    # If there is only one segment, we can simply process it
    if num_segments == 1:
        outputs = model(**tokens)
        return outputs.last_hidden_state[:, 0, :].detach()

    all_embeddings = []
    for i in range(0, len(tokens['input_ids'][0]), 512):
        batch = {k: v[:, i:i + 512] for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**batch)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].detach())

    return torch.cat(all_embeddings, dim=0)

# Function to read the file content
def read_file(file_path):
    with open(file_path, 'r', encoding='Windows-1252') as f:
        return f.read()

# Reading your two text files
text1 = read_file(r'C:\Users\DELL\Desktop\NLP_Unbias-main\NLP_Unbias-main\rt-polaritydata\rt-polarity.neg.txt')
text2 = read_file(r'C:\Users\DELL\Desktop\NLP_Unbias-main\NLP_Unbias-main\rt-polaritydata\new_rt-polarity.neg.txt')

# Getting the embeddings for your two texts using the defined function
embedding1 = embed_text(text1, model, tokenizer)
embedding2 = embed_text(text2, model, tokenizer)

# Calculating the cosine similarity between the two embeddings
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cos_sim = cos(embedding1.mean(dim=0), embedding2.mean(dim=0))

print('Cosine similarity: ', cos_sim.item())
