So contrary to what the dryad read me file says the following subjects
 (as indexed by their number on dryad) listened to twenty and journey

twenty = 3     6     7     8     9    10    13    15    17    18    19    21    24    25    27    31    32
journey =  1     2     4     5    11    12    14    16    20    22    23    26    28    29    30    33

%===========================================================================

Attached is a function I use to compute the beta weights. I've written details about the inputs and outputs.
 Also attached are the semantic impulse vectors that go as inputs. 

You will also need to reconstruct the speech envelopes using the backwards TRF as the time series data of the
 reconstructions themselves also are input to the function. You can do this with the standard backwards TRF
 approach on the cocktail party data. I've also attached the speech envelopes for both speech streams just in case. 

Also note that everything here is sampled at 64Hz

So the goal here I guess would be to estimate beta weights for the attended and unattended speech streams.
 This means dividing the subjects into 2 groups based on what they attended to (i.e. a journey and a twenty group)
 and running the function 4 times:
 Journey with stim = journey,  Journey with stim = twenty,  Twenty with stim= twenty and twenty with stim=journey.


