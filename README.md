# IR_Derry_Girls

An information retrieval (IR) and information extraction (IE) project built on the scripts of Derry Girls (Seasons 1 & 2). The project compares sparse, dense and hybrid retrieval methods for finding the right scene or line of dialogue from a natural-language query.

Implemented methods:
- Inverted index (token/word index and sentence index)
- SPLADE (sparse, transformer-based)
- BM25 (sparse, TF-IDF-based)
- FAISS (dense retrieval)
- Hybrid retrieval (BM25 + FAISS)


Evaluation

All metrics are reported at k = 5.

Method	  Precision@5	  Accuracy	  MRR  	MAP  	NDCG@5
BM25	    0.138	        0.462	      0.434	0.264	0.298
SPLADE	  0.185	        0.539	      0.368	0.218	0.297
FAISS	    0.200	        0.538	      0.462	0.350	0.397
Hybrid 
Retrieval	0.200	        0.538	      0.462	0.350	0.397

The scripts are used here for educational and research purposes only.
