from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

special_tokens_dict = {'additional_special_tokens': ['<eob>', '<eol>']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print(num_added_tokens)

# Read the text from train.en file
with open(data_path + "/MuST-Cinema/data/en-fr/train.en", "r", encoding="utf-8") as f:
    lines = f.readlines()

# lines = lines[0:100]
doc_ids = []
tokens_list = []

# Tokenize each line and store the results
for doc_idx, line in enumerate(lines):
    # Tokenize the line
    tokens = tokenizer.tokenize(line.strip())
    doc_ids.append(doc_idx + 1) #id should start from 1
    tokens_list.append(tokens)

df = (pd.DataFrame({'doc_id': doc_ids, 'token': tokens_list})
      .explode('token')
      .reset_index(drop=True)
      .groupby('doc_id')
      .apply(lambda x: x.assign(term_id=x.groupby('doc_id').cumcount() + 1))  # Assign token_id starting from 1
      .reset_index(drop=True))
      # .assign(token = lambda x: [t if t != '<eob>' else 'eob' for t in x['token']])
      # .assign(token = lambda x: [t if t != '<eol>' else 'eol' for t in x['token']]))

df.to_csv(data_path + "/MustJ/pos_bert.csv", index = False)
