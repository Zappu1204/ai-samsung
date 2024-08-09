import time
import os
import re
from urllib.parse import urlencode
import google.generativeai as genai
from openai import OpenAI
import pandas as pd
import argparse
# from langchain_groq import ChatGroq

def parse_args():
    parser = argparse.ArgumentParser(description="Standardize units")
    parser.add_argument('--method', type=str, default='ollama', choices=['langchaingroq', 'gemini', 'gpt', 'otherapi', 'ollama'], help='If you have GPU, you can choise ollama, else use api')
    parser.add_argument('--input', type=str, default='', help='Path to the input file relative path')
    parser.add_argument('--output', type=str, default='', help='Path to the output file relative path')
    parser.add_argument('--region', type=str, default='vn', choices=['vn', 'fo'], help='Get data for region medicines')
    parser.add_argument('--case', type=str, default='unit', choices=['unit', 'actIn'], help='Case of the output')
    parser.add_argument('--save', action="store_true", help='Save the output file')
    return parser.parse_args()

class Standardize:
    def __init__(self, case, client, region, content, input_path, output_path, save):
        self.case = case
        self.client = client
        self.content = content
        self.input_path = input_path
        self.output_path = output_path
        self.region = region
        self.save = save
        self.question = ''

    def strip_html_tags(text):
        clean = re.sub(r'<[^>]+>', '', text)
        return clean

    def standardize(self, value, client):
        match self.case:
            case 'unit':
                self.question = f"""Standardization of concentration and quantification "{value}". Note: You only need to return the correct standardized result, written on 1 line"""
            case 'actIn':
                self.question = f"""Convert "{value}" to chemical functional group. Note: You only need to return the correct standardized result, written on 1 line"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': self.content
                }, 
                {
                    'role': 'user',
                    'content': self.question
                }
            ], 
            temperature=0.5,
            max_tokens=256,
            model='gemma2',
        )

        match self.case:
            case 'unit':
                response = chat_completion.choices[0].message.content
            case 'actIn':
                keyword_with_tags = chat_completion.choices[0].message.content.strip()
                response = self.strip_html_tags(keyword_with_tags)

        return response
    
    def is_standar_unit(self, value):
        if not value or value == "nan" or pd.isnull(value):
            return True
        units = ["mg", "ml", "mg/ml", "%", "iu", "ui"]
        for unit in units:
            value = str(value).lower()
            if unit in value:
                return True
        return False
    
    def processing(self, client):
        df = pd.read_csv(self.input_path)
        for index, row in df.iterrows():
            match self.case:
                case 'unit':
                    if self.region == 'fo':
                        strength = row['strength'][1:-1]
                    else:
                        strength = row['strength']
                    if not self.is_standar_unit(strength):
                        standard_unit = self.standardize(strength, client)
                        if standard_unit:
                            df.at[index, 'strength'] = standard_unit
                            print(f'{strength} -> {standard_unit}')
                case 'actIn':
                    ingredient = row['englishName']
                    if not pd.isna(ingredient) and ingredient.strip():
                        standardized_ingredient = self.standardize(ingredient)
                        if standardized_ingredient:
                            df.at[index, 'Chemical_formula'] = standardized_ingredient
                            print(f"Đã chuẩn hóa: {ingredient} -> {standardized_ingredient}")

        if self.save:
            df.to_csv(self.output_path, index=False)
            print(f'Saved to: {self.output_path}')

def main():
    args = parse_args()
    region = args.region
    if region == 'fo':
        region = 'foreign'
    infile_name = f'{region}missing_form_imputation.csv'
    outfile_name = f'{region}standardized_units.csv'
    input_path = args.input
    output_path = args.output
    if os.name == 'nt':
        input_path = os.path.join(os.path.dirname(os.getcwd()), input_path, infile_name)
        output_path = os.path.join(os.path.dirname(os.getcwd()), output_path, outfile_name)
    elif os.name == 'posix':
        # input_path = os.path.join(os.getcwd(), input_path, infile_name)

        output_path = os.path.join(os.getcwd(), output_path, outfile_name)
    input_path = '/teamspace/studios/this_studio/Medicines-Data-Imputation/PreData/RAW_vn_medicines_202405171011.csv'
    
    method = args.method
    case = args.case
    # save = args.save
    # self, case, client, region, content, input_path, output_path, save
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )
    content = """Standardize the concentration and quantity from my file according to the corresponding example below.
                Example1: 1g, 1
                Standard1: 1000mg
                Example2: 0.02l
                Standard2: 20ml
                Example3: 1g/2ml
                Standard3: 500mg/ml
                Example4: ["1g"], ["3,4"]
                Standard4: 1000mg, 3400mg
                Example5: 1,0 g, 5,6 g, 2g/5ml, 1000USP
                Standard5: 1000mg, 5600mg, 400mg/ml, 1000USP
                Example6: "."
                Standard6: .
                Other units such as %, iu, ui or "no unit" will remain the same,
                If the input is a strange character or cannot be normalized, return blank "".
                The output only needs the normalized part, not including any information, you don't need text or anything to describe the way you work.
    """
    content2 = """You are an AI assistant capable of analyzing drug information. Your task is to read an input text string describing the name of a drug or its active ingredient, then separate this information into Chemical formula .
                For example:
                Input: "Calcium gluconate 1g effervescent tablets"
                Output: C12H22CaO14
                Follow these rules:
                Chemical formula:Priority is given to converting drug names or active ingredients into chemical formulas if found corresponding, only write the chemical formula on 1 line like for example "C5H9NO3S". If no chemical formula is found, keep the original name. If there is only "[]" then return blank cells.
                Please note that if the formula is not found, it returns empty
                Error handling:If the input is an odd character or cannot be normalized, return blank "".
                Now apply the above rules to process the following input:"""

    content3 = """You are an AI assistant capable of analyzing drug information. Your task is to read an input text string describing the name of a drug or its active ingredient, then separate this information into a chemical formula.
                For example:
                * Input: ".acetylcysteine"
                * Output: C5H9NO3S
                Follow these rules:
                Chemical formula: Priority is given to converting drug names or active ingredients into chemical formulas if found corresponding, only write the chemical formula on 1 line like for example "C5H9NO3S". If no chemical formula is found, keep the original name. If there is only "[]" then return blank cells.
                Error handling: If the input is an odd character or cannot be normalized, return blank "".
                Now apply the above rules to process the following input:"""

    content4 = """You are an AI assistant capable of analyzing drug information. Your task is to read an input text string describing the name of a drug or its active ingredient, then separate this information into a chemical formula."""
    match case, method:
        case 'unit', 'ollama':
            standardize_units = Standardize(case, region, client, content, input_path, output_path)
            standardize_units.processing(client)
        case 'actIn', 'ollama':
            standardize_actin = Standardize(case, region, client, content, input_path, output_path)
            standardize_actin.processing()
    
if __name__ == '__main__':
    main()


   