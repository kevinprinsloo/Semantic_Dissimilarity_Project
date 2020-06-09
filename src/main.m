clear all;
runs=1:20;

load('E:\Workspace 2019\Semantic TRF - Methods\GloVe\glove42B_300d_small');        %Loads in the vectors for all words (i.e. word2vec or glove)
load('E:\Workspace 2019\Misc\General Functions\functionWords');     %Load in a list of function words
Fs=64;

load('C:\Users\kevin\Documents\Semantic_Dissimilarity\Cocktail_Party\Stimuli\glove42B_300d_small.mat')
load('C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src\funcWords.mat')

for r = 1:30
    
    filename=(['C:\Users\kevin\Documents\Semantic_Dissimilarity\Cocktail_Party\Stimuli\Text\20000\Run' int2str(r) '.mat']); %Reads the text file of the trial in question
    [words, semanticDissim,wordVectors]=calculateSemanticDissim(filename, funcWords, conceptRowsC,datM2); 
    filename=['.\wordOnsets\phonemes' int2str(r) '.txt'];   % Reads the word onset file
    [semVectors{r},onsetVectors{r}] =createTimeVectors(filename,words,semanticDissim,64);        
    
end

