clear all;
runs=1:20;
cd('E:\Workspace 2019\Semantic TRF - Methods\Stimuli');
load('E:\Workspace 2019\Semantic TRF - Methods\GloVe\glove42B_300d_small');        %Loads in the vectors for all words (i.e. word2vec or glove)
load('E:\Workspace 2019\Misc\General Functions\functionWords');     %Load in a list of function words
Fs=128;

for r=runs
    
    filename=['.\Text\text' int2str(r) '.txt']; %Reads the text file of the trial in question
    [words, semanticDissim,wordVectors]=calculateSemanticDissim(filename, funcWords, conceptRowsC,datM2); 
    filename=['.\wordOnsets\phonemes' int2str(r) '.txt'];   % Reads the word onset file
    [semVectors{r},onsetVectors{r}] =createTimeVectors(filename,words,semanticDissim,128);
        
    
end

