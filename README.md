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



Re-attribute the summery pieces to individual conversations

Development was made over CPU based laptop hence using pickle intensively to save intermediate results/embeddings

This challenge could be addressed as a retrieval or classification challenge, I’ve started exploring the retrieval one but described the former as well. 

Retrieval 
Retrieve the dialog from the queried chunks of the (absent) summary by using semantic textual similarity. Assuming a chunk should be similar semantically to the dialog it has been taken. This assumption is rough and will be discussed later. The many-to-many matching optimization problem based on a semantic similarity score between each chunk to each dialog could be addressed by (minimum) linear assignment optimized by the Hungarian algorithm (see appendix), which is sub-optimal (related to brute force) for this polynomial complexity assignment. 

A textual similarity model such as SBERT (Bi-Encoder) was trained to get maximal cosine similarity for semantic similar sentences. 

Disclaimer: there are many SBERT models under various challenges such as MTEB one. Hence picking the right model is also an issue to be further optimized.

I've chosen one of the top models, mostly used, in the leaderboard, all-MiniLM-L6-v2, which yields an embedding size of 384. It supports a sequence length of 256 tokens (WordPiece) which is roughly equivalent to 150 words. Cosine similarity is taken between SBERT embeddings of each chunk to dialog segment.

Hence the retrieval challenge could be addressed by “linear matching” between chunks and dialog segmented, up to 150 words each. I’ve removed the LF, CR after using them as delimiters for the segmentation and added spaces instead.

Since the dialog is segmented as well as the summary, the order of the matched chunks is given by construction. That is to say, a chunk will be associated with the right dialog segment in the right order.

The optimization is taken over the cost function matrix, C of [n_chunks x m_dialog_segments] 1287x3910 in the training set.


Classification 
Zero-shot: assessing the P(x_chunk/summ_conversation_of_y), namely project that problem to single dimension problem by the proxy of cosine similarity over the contextualized dialog and chunk embeddings. Of course it is less optimal than global optimization treatment as in the retrival.

One of the options is solving the order of the chunks within the summary, is by maximizing the pairwise likelihood of the next sentence prediction, using BERT for instance. This approach lacks the original dialog information.

From the training data determine the min threshold of cosine similarity of the conversation segments  embeddings to the chunk taken out of it. 

Increase the dimension of the by using cross encoder. Taking all the permutations of concatenated chunks and conversation segments will create embeddings of 512 per each concatenate that to 1024 and train a classifier with more parameters: the dimension space has been increased, to better determine the matching. 

Method 
Split reference data into train and validation
Evaluate POC under the training set, adjust hyperparameters, validate over the validation set and yield results over the test set


Creating summarization data
Assuming a chunk should be semantically similar to its originating dialog (segment) isn’t perfect. Since the summarization chunks resulted from a larger context than a dialog segment.

Hence, another option is to use a summarizer (GPT) to create, a near-original, summary data to be more similar to the queried chunks as a proxy. Then try to match to the generated summary instead.

Creating summarization data for improved retrieval 
The following approach can lead to the summarization text which may be expected to be more similar to the query and by that increase the recall.

As an example: the following summary of the ChatGPT prompt:

Sam excitedly reveals to Tina that she had a date the previous night through a dating app. Despite a comical start where Sam didn't recognize her date initially, they had a good time, and Sam believes he might be "the one," leading to playful banter with Tina about planning a wedding.







Report and error analysis
Using the training data for assignment of chunk to dialog segments.

Recall - defined as the ratio of the right assignment/attribution of chunks to dialog with the right order related to the total dialogues   

Training set
Recall of assigning chunk to the right dialog was 0.797 (order maintained built-in)
