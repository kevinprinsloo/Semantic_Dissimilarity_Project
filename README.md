# Semantic_Dissimilarity_Project
 
### Websites
GloVe: Global Vectors for Word Representation: https://nlp.stanford.edu/projects/glove/<br/> 
Dataset Dryad: https://datadryad.org/stash/landing/show?big=showme&id=doi%3A10.5061%2Fdryad.070jc<br/>
<br/>
### Peripheral Notes
So contrary to what the dryad read me file says the following subjects<br/> 
 (as indexed by their number on dryad) listened to twenty and journey<br/> 
<br/> 
twenty = 3     6     7     8     9    10    13    15    17    18    19    21    24    25    27    31    32<br/> 
journey =  1     2     4     5    11    12    14    16    20    22    23    26    28    29    30    33<br/> 
<br/> 
%===========================================================================<br/> 
<br/> 
### Task
reconstruct the speech envelopes using the backwards TRF as the time series data of the<br/> 
reconstructions themselves also are input to the function. You can do this with the standard backwards TRF<br/> 
approach on the cocktail party data. I've also attached the speech envelopes for both speech streams just in case.<br/>  
(Also note that everything here is sampled at 64Hz)<br/> 
<br/> 
So the goal here I guess would be to estimate beta weights for the attended and unattended speech streams.<br/> 
 This means dividing the subjects into 2 groups based on what they attended to (i.e. a journey and a twenty group)<br/> 
 and running the function 4 times:<br/> 
 Journey with stim = journey,  Journey with stim = twenty,  Twenty with stim= twenty and twenty with stim=journey.<br/> 
 
 ## Cocktail Party Experiment

Files
eegData: EEG Data, Time Locked to the onset of the speech stimulus.   
Format: Channels (128) x Time Points

mastoids: Mastoid Channels, Time Locked to the onset of the speech stimulus. 
Format: Channels (Left=1 Right=2) x Time Points

fs: Sampling Rate 

EEG data is unfiltered, unreferenced and sampled at 128Hz

Experiment Information
Subjects 1-17 were instructed to attend to 'Twenty Thousand Leagues Under the Sea' (20000), played in the left ear
Subjects 18-33 were instructed to attend to 'Journey to the Centre of the Earth' (Journey), played in the right ear
