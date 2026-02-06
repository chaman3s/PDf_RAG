from Embedding import TextEmbedding as te 

if __name__ == "__main__": 
    t = te.textEmbedding()
    s=["fghjkcvbnmdfghjs"]
    emb = t.txtToEmbedding(s)
    print(emb)