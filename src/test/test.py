import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_recipes_from_txt(directory):
    recipes = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                title = lines[0].strip()
                ingredients_index = lines.index("材料:\n") + 1
                instructions_index = lines.index("作り方:\n") + 1
                categories_index = lines.index("カテゴリ:\n") + 1
                ingredients = lines[ingredients_index:instructions_index-1]
                instructions = lines[instructions_index:categories_index-1]
                categories = lines[categories_index:]

                recipes.append({
                    'title': title,
                    'ingredients': ', '.join([line.strip() for line in ingredients]),
                    'instructions': ' '.join([line.strip() for line in instructions]),
                    'categories': ', '.join([line.strip() for line in categories]),
                    'description': f"{title} - {''.join([line.strip() for line in instructions])}"
                })
    return pd.DataFrame(recipes)

def search_recipes(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.reshape(1, -1), recipe_embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    return recipes_df.iloc[top_k_indices]

def main():
    while True:
        query = input("レシピを検索するためのキーワードを入力してください（終了するには'quit'と入力）：")
        if query.lower() == 'quit':
            break
        results = search_recipes(query)
        print("検索結果:")
        for idx, recipe in results.iterrows():
            print(f"タイトル: {recipe['title']}")
            print(f"材料: {recipe['ingredients']}")
            print(f"作り方: {recipe['instructions']}")
            print(f"カテゴリ: {recipe['categories']}")
            print("---------------")

recipes_df = load_recipes_from_txt('./rakuten_recipe')

# sentenceBERTモデルのロード
model = SentenceTransformer('tohoku-nlp/bert-base-japanese-char-whole-word-masking')

# レシピ説明文の埋め込みを生成
recipe_embeddings = model.encode(recipes_df['description'].tolist(), convert_to_tensor=True)

main()