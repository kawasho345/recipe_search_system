import torch
from sentence_transformers import SentenceTransformer
from transformers import BertJapaneseTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_sentences = [
    "ダーウィンの進化論は嘘です",
    "バナナはおやつに入ります",
    "地球は平面です",
    "日本では応用数学科はまだまだ少ない方です",
    "数学と数理科学の違いはなんですか",
    "散歩をするとバナナがおやつに入らないかもしれないことに気づきました",
    "地球の半径を数理的に求めることができます",
    "私はバナナが好きな数学者です",
    "数学とバナナは同じです",
    "残念ながら、地球とバナナとゴリラは同じではありません",   #10
    "ダーウィンはバナナをおやつと考えました",
    "これ以上無意味な文章を作ることをやめませんか",
    "数理の世界は長い年月を経て進歩してきましたが、人間は長い年月を経てゴリラに近づきました",
    "ダーウィンは進化論の提唱者ですが、ダテミキオはカロリー0理論の提唱者です",
    "その理論を応用することで、バナナを用いてブラックホールを生成する方法を数学的に導くことができます",
    "ピザはその高さを0に近づけることで体積が0に近づくためカロリーは0",
    "ダーウィンはゴリラの進化元です",
    "バナナのカロリーは1本86キロカロリーです",
    "どうして地球にはピザが自生していないのですか",
    "ここまでだいたい嘘"   #20
]

sentences_dataframe = pd.DataFrame(all_sentences)

model_path = "model/sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

model = SentenceTransformer(model_path, device=device)

all_embeddings = model.encode(all_sentences)

# tokenizer_cl_tohoku = BertJapaneseTokenizer.from_pretrained(model_path)

# all_tokens_cl_tohoku = [tokenizer_cl_tohoku.tokenize(sentence) for sentence in all_sentences]

def search_recipes(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity(query_embedding.reshape(1, -1), all_embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    return sentences_dataframe.iloc[top_k_indices]

def main():
    while True:
        query = input("レシピを検索するためのキーワードを入力してください（終了するには'quit'と入力）：")
        if query.lower() == 'quit':
            break
        results = search_recipes(query)
        print("検索結果:")
        for idx, recipe in results.iterrows():
            print(recipe)

main()