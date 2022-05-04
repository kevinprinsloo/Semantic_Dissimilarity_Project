# Semantic_Dissimilarity_Project

![This is an image](https://github.com/kevinprinsloo/Semantic_Dissimilarity_Project/blob/master/Semantic_decoding_methods_size.jpg)

### Introduction

There is continued debate on whether context-based predictions of upcoming words feed back to affect the acoustic processing of speech, especially in the context of natural, continuous speech.

Recently, using narrative speech, our group showed that the more semantically similar a word is to its preceding context, the better it is encoded at acoustic-phonetic levels, providing evidence of an influence of top-down feedback.

Here, we aim to replicate this finding in the context of a cocktail-party attention experiment. 

Doing so would allows us to confirm that our result is genuinely driven by top-down predictive activity and not by any confounding acoustic-linguistic correlations.

The more semantically similar a word is to its preceding context, the better it is encoded at the acoustic-phonetic levels, providing evidence for an influence of top-down feedback

We explored this using a cocktail party attention paradigm, where participants must attend to one of two competing speakers. Stimulus were natural continuous speech.
Using computational modeling we were able to investigate this in a two stage processes.

Stage 1: first we represented speech in its lower-level representations, the speech amplitude envelope. And at a higher level we parametrised speech in terms of its surprisal and semantic similarity using computational language models
Stage 2: we used the word level envelope reconstructions accuracies and passed it on to stage 2 and regressed against semantic similarity and surprisal.

### Websites
GloVe: Global Vectors for Word Representation: https://nlp.stanford.edu/projects/glove/<br/> 
Dataset Dryad: https://datadryad.org/stash/landing/show?big=showme&id=doi%3A10.5061%2Fdryad.070jc<br/>
<br/>
### Peripheral Notes
So contrary to what the dryad readme file says the following subjects<br/> 
 (as indexed by their number on dryad) listened to twenty and journey<br/> 
 
