import pandas as pd
import psycopg2
import json
import re

conn = psycopg2.connect(
    host="177.177.0.100",
    dbname='DatasetDB',
    user="imst",
    password="1qaz2wsx-"
)
#csv dosya kaydedilecek adres
path=r" "
table_name = "endpoint_dataset"

df = pd.read_sql(f"SELECT question, answer FROM {table_name}", conn)
conn.close()
json_lines = []

for _, row in df.iterrows():
    question = row['question'].strip()
    answer = row['answer'].strip()

    # "Kullanıcı mesajı:" kısmını temizleme
    question = re.sub(r'^Kullanıcı mesajı:\s*', '', question)

    json_object = [
        {
            "from": "human",
            "value": question
        },
        {
            "from": "gpt",
            "value": answer
        }
    ]
    json_lines.append(json.dumps(json_object, ensure_ascii=False))

formatted_df = pd.DataFrame({'conversations': json_lines})
formatted_df.to_csv(path, index=False, sep=';', encoding='utf-8-sig')