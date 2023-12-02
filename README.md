# chunks_to_conversations
Settings

First, compute embeddings  
pre_compute_embeddings = True
Set the right flag train_data=True or False for that 

For running the prediction process, based on the precomputed embeddings, set:
pre_compute_embeddings = False
and set the right flag train_data=True or False related to what data used for the prediction

Experimental - convert abbreviations and remove emojis
Setting: 
clean_noisy_segment = True
and rerun the pre_compute_embeddings (=True) to recompute embeddings based on processed dialog


