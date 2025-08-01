import pandas as pd
import os

method_name_dct = {
    'benign':['benign'],
    'toxic':['violent','political','sexual','disturbing']
}
prompt_class={
    'benign':0,'violent':1,'political':2,'sexual':3,'disturbing':4
}
    
def trim_quotes(s):
    return s.strip("\"'")

'''Used to process spaces and punctuation in text to make it more standardized'''
def process_spaces(prompt):
    prompt=prompt.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()
    return trim_quotes(prompt)

def load_MyData(file_folder=None):
    data={
        'train':[],
        'test':[],
        'valid':[]
    }
    folder=os.listdir(file_folder)
    for now in folder:
        if now[-3:]!='csv':
            continue
        full_path=os.path.join(file_folder,now)
        keyname=now.split('.')[0]
        assert keyname in data.keys(), f'{keyname} is not in data.keys()'
        now_data=pd.read_csv(full_path,on_bad_lines='skip')
        for i in range(len(now_data)):
            id=now_data.iloc[i]['id']
            prompt,src=now_data.iloc[i]['prompt'],now_data.iloc[i]['category']
            label= '1' if src=='benign' else '0'
            '''The label represents whether it is good or not, and the catagery_id represents the category'''
            data[keyname].append((process_spaces(str(prompt)),label,src,id))
    
    return data
