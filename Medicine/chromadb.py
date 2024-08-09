import os, re, json, random, argparse
import chromadb
from tqdm import tqdm
import pandas as pd

# import langchain
# from langchain.vectorstores import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description='Create vector store')
    parser.add_argument('--input', type=str, default='PreData/drug.csv', help='Path to the input file')
    parser.add_argument('--output', type=str, default='database/active_ingredient_vector_store', help='Path to the output file')
    parser.add_argument('--collection_name', type=str, default='active_ingredient', help='Name of the collection')
    parser.add_argument('--case', type=str, default='createdb', choices=['createdb', 'query'], help='Case of the output')
    return parser.parse_args()

def standardize_response(query, context):
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',  # Bắt buộc nhưng không sử dụng
    )
    chat_completion = client.chat.completions.create(
        messages=[
            # {
            #     'role': 'system',
            #     'content': 
            # },
            {
                'role': 'user',
                'content': f""" Context: {context}
                                Query: {query}
                            """
            }
        ],
        temperature=0.5,
        top_p=0.3,
        max_tokens=64,

        model='gemma2',
    )
    
    keyword_with_tags = chat_completion.choices[0].message.content.strip()
    key_clean = re.sub(r'<[^>]+>', '', keyword_with_tags)
    return key_clean


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), input_path)
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_path)
    collection_name = args.collection_name
    case = args.case

    match case:
        case 'createdb':
            documents = []
            ids = []
            metadatas = []
            resource = r'data/drug92.csv'
            resource = os.path.join(os.path.dirname(os.path.dirname(__file__)), resource)

            df = pd.read_csv(resource)
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing CSV"):
                dic = {
                    "compound name": row['cmpdname'],
                    "compound synonym": row['cmpdsynonym'],
                    "molecular formula(chemical formula)": row['mf'],
                    "iupacname": row['iupacname'],
                }
                documents.append(dic)
                ids.append(row['cid'])
                metadatas.append(row['inchi'])
            
            client = chromadb.PersistentClient(path=output_path)
            print("Creating embedding function")
            ollama_ef = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                # api_key="ollama",
                model_name="gemma2"
            )
            print("Creating collection")
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=ollama_ef
            )
            print("Adding to ChromaDB")
            with tqdm(total=1, desc="Adding to ChromaDB") as pbar:
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
                pbar.update(1)
            
            print(f"Collection {collection_name} created successfully")
            
        case 'query':
            client = chromadb.PersistentClient(path=input_path)

            ollama_ef = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                api_key="ollama",
                model_name="gemma2",
                task_type="RETRIEVAL_QUERY"
            )                
            collection = client.get_collection(
                name=collection_name,
                embedding_function=ollama_ef
            )

            query = """ You are tasked with converting active ingredient names into their corresponding molecular formulas based solely on the provided context. If the context does not provide enough information or if you are uncertain, respond with 'None'. Do not write any subscripted numbers.Do not provide any explanations or unrelated information.
                        Examples:
                        ["Calcium gluconate"] -> C12H22CaO14
                        ["Warfarin"] -> C19H16O4
                        ["Tolvaptan"] -> C26H25ClN2O3
                        Sắt (III) hydroxyd polymaltose 34% -> C12H25FeO
                        Desired Format:(Provide the full chemical formula on a single line, ensuring that all characters are the same size. If there are multiple chemical formulas, list them on one line separated by commas (",").) """

            df = pd.read_csv(input_path)
            for index, row in df.iterrows():
                ingredient = row['activeIngredient']
                if not pd.isna(ingredient) and ingredient.strip():
                    results = collection.query(
                        query_texts=[query + ingredient],
                        n_results=5,
                        include=["documents", "metadatas"]
                    )
                    response = standardize_response()
    

if __name__ == '__main__':
    main()