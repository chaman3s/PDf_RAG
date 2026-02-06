from Embedding import TextEmbedding as te 
from pdf_pipeline import ingest as pdf
from pdf_pipeline import storage as sav
import json

if __name__ == "__main__": 
    # pdf.main()
    with open("pdf_pipeline/output/parsed_output.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        print("data:",data['raw_pages'][0])
        t= te.textEmbedding()
        result=[]
        for r in data['raw_pages']:
            print(f"---p-- {r["page"]}")
            emt=t.TxtToEmbedding(r[ 'content']).tolist()
            result.append({"page": r["page"],"emb": emt,"content": r["content"]})
        sav.save_json(result, "output/result.json")
            
            
            
            
   