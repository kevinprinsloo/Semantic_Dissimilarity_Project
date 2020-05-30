% Summary:
% Script to run mTRF decoding/encoding

% Status:
% Under Development

% Notes:
% n/a

% Author(s):
% Kevin Prinsloo

% Editor(s):
%

%% Prepare Workspace
%clearvars
%close all
%clc


% Manually Initialise Variables
% addpath  '/scratch/kprinslo/Semantic_Dissimilarity/Cocktail_Party/mTRF_KP_edit/';
% addpath  '/scratch/kprinslo/Semantic_Dissimilarity/Cocktail_Party/Resources_Misc/';
% addpath  '/scratch/kprinslo/Semantic_Dissimilarity/Cocktail_Party/Scripts/';
% data_path = '/scratch/kprinslo/Semantic_Dissimilarity';
% git_path = '/scratch/kprinslo/Semantic_Dissimilarity';
% pc_path = '/scratch/kprinslo/Semantic_Dissimilarity';

addpath 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src\Prosody_scr';
addpath 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src\';
%rmpath(genpath('E:\Semantic_Dissimilarity\Cocktail_Party\Prosody_measures\AcousticStats_Sam\'));
%rmpath(genpath('E:\Semantic_Dissimilarity\Cocktail_Party\Prosody_measures\'));

data_path = 'C:\Users\kevin\Documents\Semantic_Dissimilarity';
addpath('C:\Users\kevin\Documents\Github\Filtering');
addpath(genpath('C:\Users\kevin\Documents\Semantic_Dissimilarity\Cocktail_Party\Prosody_measures\AcousticStats_Sam\'));
addpath(genpath('C:\Users\kevin\Documents\Semantic_Dissimilarity\Cocktail_Party\Prosody_measures\'));

addpath 'C:\Users\kevin\Chimera_Study_Desktop\Toolboxes\fieldtrip-20200130\fieldtrip-20200130';
ft_defaults
eeglab
close all
addpath 'C:\Users\kevin\Chimera_Study_Desktop\Toolboxes\fieldtrip-20200130\fieldtrip-20200130'
ft_defaults

%% Add subfolder/dir
%addpath 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src'
%addpath 'C:\Users\kevin\Documents\Github\mTRF_KP_edit\'
%data_path = 'E:\Semantic_Dissimilarity';
%git_path = 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project';
%pc_path = 'C:\Users\kevin\Documents\Semantic_Dissimilarity';

% Define study folder
study_name = 'Cocktail_Party';

% Initialise EEG Variables
eeg_sampling_rate_original_Hz = 128;
channels_number_cephalic = 128;
eeg_sampling_rate_downsampled_Hz = 64;

% Define conditions
conditions = {'20000','Journey'};

%% Filter setup
% Lowpass filter
Fpass = 3.6e3;
Fstop = 4.8e3;
Fs=44100;
Apass = 1;
Astop = 60;
h = fdesign.lowpass(Fpass,Fstop,Apass,Astop,Fs);
lpf1 = design(h,'cheby2','MatchExactly','stopband');
clear Fpass Fstop Fs h;

wav_fs=44100; % Sampling frequency of source audio files
down_factor=5; % envelope downsampling factor
EEG_fs=64;
eye_data_fs=500;
[P,Q] = rat((EEG_fs*down_factor)/wav_fs);
[Peye,Qeye] = rat((eye_data_fs*down_factor)/wav_fs);

% GammacHirp filterbank
GCparam.fs = wav_fs/down_factor;
GCparam.NumCh = 256;
GCparam.FRange = [80,3e3];
GCparam.OutMidCrct = 'ELC';
% GCparam.OutMidCrct = 'No';
% GCparam.Ctrl = 'dyn';

for condition_idx = 1
    condition_name = conditions{condition_idx};
    
    % List .wav files
    listing = dir(fullfile([data_path,'/',study_name,'/','Stimuli','/','Wav_files','/',condition_name,'/'],'*.wav'));
    stim_listings = {listing.name};
    stim_listings = natsortfiles(stim_listings); % Correction for numerical sorting
    
    for stim_idx = 1:length(stim_listings)
        
        file_name = [stim_listings{stim_idx}];
        file_directory = [data_path,'/',study_name,'/','Stimuli','/','Wav_files','/',condition_name,'/'];
        
        % Load in .wav file.
        [y, wav_fs] = audioread([data_path,'/',study_name,'/','Stimuli','/','Wav_files','/',condition_name,'/',...
            stim_listings{stim_idx}]);
        
        % Compute f0
        eeg_fs = 64;
        [f0, t] = get_f0_praat_sm(y', wav_fs, eeg_fs);
        
        % Compute resolvability
        [f_rel, res] = measure_prosody_sm(file_name, file_directory, eeg_fs);
        
        close all
        
        %% Prosody information
        % Verify Directory Exists and if Not Create It
        if exist([data_path,'/',study_name,'/','Stimuli','/','Prosody_variables','/',condition_name,'/'],'dir') == 0
            mkdir([data_path,'/',study_name,'/','Stimuli','/','Prosody_variables','/',condition_name,'/']);
        end
        % Save Figures and Data
        filename =[data_path,'/',study_name,'/','Stimuli','/','Prosody_variables','/',condition_name,'/',...
             'stim_trial_',num2str(stim_idx)]; filetype = '.mat';
        save([filename,filetype],'f0','f_rel','res','-v7.3'); clear filename filetype
        fprintf('Saving');
        clear f0 f_rel res
        
        %% >> Apply Gammatone filter bank
        % Filter below Nyquist frequency
        y = filtfilthd(lpf1,y);
        y=y';
        
        % Cochlear Bandpass filtering
        spectrogram = GCFBv210(y,GCparam); clear y
        
        % Calculate narrowband and broadband envelopes
        for chn=1:size(spectrogram,1)
            spectrogram(chn,:)=abs(hilbert(spectrogram(chn,:)));
        end
        envelope = mean(spectrogram,1);
        clear spectrogram
        
        % Downsample
        P = 64;
        Q = wav_fs;
        envelope = resample(envelope,P,Q); % change these values depending on whether envelope is wanted for relating to EEG or eye tracking
        
        %% Save Gammatone Envelopes
        % Verify Directory Exists and if Not Create It
        if exist([data_path,'/',study_name,'/','Stimuli','/','Envelopes_Gammatone','/',condition_name,'/'],'dir') == 0
            mkdir([data_path,'/',study_name,'/','Stimuli','/','Envelopes_Gammatone','/',condition_name,'/']);
        end
        % Save Figures and Data
        filename =[data_path,'/',study_name,'/','Stimuli','/','Envelopes_Gammatone','/',condition_name,'/',...
            stim_listings{stim_idx}]; filetype = '.mat';
        save([filename,filetype],'envelope','-v7.3'); clear filename filetype envelope
        fprintf('Saving\n');
        
    end
end




