
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

addpath 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src'
addpath 'C:\Users\kevin\Documents\Github\mTRF_KP_edit\'

% Manually Initialise Variables

%% Add subfolder/dir
data_path = 'E:\Semantic_Dissimilarity';
git_path = 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project';

% Define study folder
study_name = 'Cocktail_Party';

% Initialise Subject Variables
listing = dir([data_path,'/',study_name,'/','EEG_Data','/']);
subejct_listings = {listing.name};
subejct_listings(cellfun('length',subejct_listings)<3) = [];
subejct_listings(end) = [];
subjects_orig = subejct_listings;
subjects_number = numel(subjects_orig);
subjects = natsortfiles(subjects_orig); % Correction for numerical sorting

% Initialise EEG Variables
eeg_sampling_rate_original_Hz = 128;
Fs = eeg_sampling_rate_original_Hz;
channels_number_cephalic = 128;
eeg_sampling_rate_downsampled_Hz = 64;

% Define conditions
conditions = {'20000','Journey'};

% Define mTRF paramters
mapping = -1; % Backwards [-1] | Forwards [1]
start = -200;
fin = 800;
singleLags = unique([start:5:fin,0:5:500]);
lambda_test_values = 2.^(2:2:30);
lambda_value_plotting = 1e3; % 1e2
multivariate_dimensions_number = 1; % define if multivariate (define n) or univariate (=1);
baseline_correction_TRF = 'BC'; % NA - for no BL TRF correction
fs = 128;
dir = -1;
epoch_higher_cutoff_SNR_ms = 600;
epoch_low_cutoff_SNR_ms = 0;

%% Preprocess data
for subject_idx = 1:length(subjects)
    
    listing = dir([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/','*.mat']);
    trial_listings = {listing.name};
    trial_listings = natsortfiles(trial_listings); % Correction for numerical sorting
    
    % Load EEG Data
    clear eeg_holder
    for trial_idx = 1:length(trial_listings)
        load([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/',trial_listings{trial_idx}],'eegData');
        eegData=double(eegData); % Data was single matrix when downloaded - need double for preprocessing
        
        % Check data dimentions consistent #>> ToDo check why some flipped
        [dim1,dim2] = size(eegData);
        if dim1 == 128; eegData = eegData'; end
        
        %% Filter EEG Data
        Fstop1 = 0.1; % Lower stopband freq (Hz)
        Fpass1 = 0.2; % Lower passband freq (Hz)
        Fpass2 = 30;  % Upper passband freq (Hz)
        Fstop2 = 32;  % Upper stopband freq (Hz)
        Astop  = 60;  % Stopband attenuation (dB)
        Apass  = 1;   % Passband attenuation (dB)
        
        % Generate bandpass filter
        h = fdesign.highpass(Fstop1,Fpass1,Astop,Apass,Fs);
        hpf = design(h,'cheby2','MatchExactly','stopband'); clear h
        l = fdesign.lowpass(Fpass2,Fstop2,Apass,Astop,Fs);
        lpf = design(l,'cheby2','MatchExactly','stopband'); clear i
        % Apply filter
        flteeg = filtfilthd(hpf,eegData); clear eegData
        flteeg = filtfilthd(lpf,flteeg); clear hpf lpf
        
        %% Downsample Data
        for chanIdx = 1:size(flteeg,2)
            eeg_temp(:,chanIdx) = resample(flteeg(:,chanIdx),eeg_sampling_rate_downsampled_Hz,eeg_sampling_rate_original_Hz); %#ok<*SAGROW>
        end
        eeg_trial = eeg_temp; clear eeg_temp
        
        % Concatinate all epochs
        eeg_holder(trial_idx,:,:) = eeg_trial; clear eeg_trial
    end
    
    %% Remove bad channels
    channels_number_cephalic = size(eeg_holder,3);
    channel_locations = readlocs([git_path,'/','Resources_Misc','/','BioSemi','_',num2str(channels_number_cephalic),'_','AB','.sfp'],'filetype','sfp');
    
    % Put data into EEGLab data structure
    eeg_holder = permute(eeg_holder,[3 2 1]); % prepare for EEGLab structure
    clear EEG
    EEG.data = eeg_holder; EEG.nbchan = channels_number_cephalic;
    EEG.srate = 512;
    EEG.chanlocs = channel_locations;
    EEG.trials = size(eeg_holder,3);
    EEG.pnts = size(eeg_holder,2);
    EEG.xmin = 0; EEG.xmax = (size(eeg_holder,2)/64)*1000;
    EEG.icaact = []; EEG.comments = '';
    EEG.epoch = []; EEG.setname = '';
    EEG.filename = ''; EEG.filepath = '';
    EEG.subject = ''; EEG.group = '';
    EEG.condition = ''; EEG.session = [];
    EEG.times = ''; % dataX_epoch.time{1};
    EEG.ref = []; EEG.event = [];
    EEG.icawinv = []; EEG.icasphere = [];
    EEG.icaweights = []; EEG.icaact = [];
    EEG.saved = 'no'; EEG.etc = [];
    EEG.specdata = []; EEG.icachansind = []; EEG.specicaact = [];
    clear eeg_holder
    
    % Select bad channel removal thresholds
    [~,idx1] = pop_rejchan(EEG,'elec',1:channels_number_cephalic,'threshold',4,...
        'norm','on','measure','kurt');
    [~,idx2] = pop_rejchan(EEG,'elec',1:channels_number_cephalic,'threshold',4,...
        'norm','on','measure','prob');
    [~,idx3] = pop_rejchan(EEG,'elec',1:channels_number_cephalic,'threshold',4,...
        'norm','on','measure','spec');
    badChans = unique([idx1,idx2,idx3]);
    %EEG = pop_select( EEG,'nochannel',badChans); % KP
    
    % Spline interpolate bad channels
    if ~isempty(badChans)
        EEG = pop_interp(EEG,badChans,'spherical');
    end
    eeg_data = double(EEG.data); %>> TODO check why it was converted back to signal matrix
    
    %% Re-reference data
    eeg_trial_clean = zeros(size(eeg_data));
    for k = 1:size(eeg_data,3)
        eeg_trial = eeg_data(:,:,k);
        dat = ft_preproc_rereference(eeg_trial, 'all', 'avg'); clear eeg_trial
        %dat = ft_preproc_rereference(eeg_trial, [129 130], 'avg'); clear eeg_trial
        %dat = ft_preproc_detrend(dat);
        eeg_trial_clean(:,:,k) = dat; clear dat
    end
    clear eeg_data
    
    for trial_idx = 1:length(trial_listings)
        
        % Extract epoch
        eeg_trial = eeg_trial_clean(:,:,trial_idx);
        
        %% Save ReRef
        % Verify Directory Exists and if Not Create It
        if exist([data_path,'/',study_name,'/','Recordings','/',subjects{subject_idx},'/'],'dir') == 0
            mkdir([data_path,'/',study_name,'/','Recordings','/',subjects{subject_idx},'/']);
        end
        % Save Figures and Data
        filename = [data_path,'/',study_name,'/','Recordings','/',subjects{subject_idx},'/',...
            trial_listings{trial_idx}];
        save(filename,'eeg_trial','-v7.3'); clear filename filetype
        clear eeg_trial
    end
end

%% mTRF Analysis
for subject_idx = 1:length(subjects)
    for condition_idx = 1:length(conditions)
        condition_name = conditions{condition_idx};
        
        % Preallocation of variables
        stim = cell(1,length(goodtrials));
        resp = cell(1,length(goodtrials));
        
        listing = dir([data_path,'/',study_name,'/','Recordings','/',num2str(subjects{subject_idx}),'/','*.mat']);
        trial_listings = {listing.name};
        trial_listings = natsortfiles(trial_listings); % Correction for numerical sorting
        
        % Load EEG Data
        stim = cell(1,length(trial_listings));
        resp = cell(1,length(trial_listings));
        cross_validation_idx = 1;
        for trial_idx = 1:length(trial_listings)
            
            %% >> Load EEG data
            load([data_path,'/',study_name,'/','Recordings','/',subjects{subject_idx},'/',...
                trial_listings{trial_idx}],'eeg_trial');
            
            % Z-score EEG
            eeg_trial = eeg_trial';
            eeg_trial = zscore(eeg_trial(:,1:128));
            
            % Deal with missing trials - index which stim to load that matches EEG
            str = trial_listings{trial_idx};
            str = str(9:end-4); stim_trial = string(regexp(str,'\d*','Match'));
            
            % Load modulating signal
            load([data_path,'/',study_name,'/','Stimuli','/','Envelopes','/',condition_name,'/',...
                stim_listings{trial_idx}],'envelope');
            
            % Convert Gamma tone filterbank envelope to dB SPL
            clear modulating_signal_voltage modulating_signal_voltage_temp modulating_signal_SPL_from_voltage modulating_signal_holder
            clear  modulating_signal_holder2  modulating_signal_holder3
            modulating_signal_voltage = envelope; %#ok<*SAGROW>
            modulating_signal_voltage = max(0,modulating_signal_voltage)/max(max(0,modulating_signal_voltage));
            modulating_signal_voltage_temp = 10000*(((1-(1/10000))*modulating_signal_voltage)+(1/10000));
            modulating_signal_SPL_from_voltage = 20*log10(modulating_signal_voltage_temp); clear modulating_signal_voltage_temp
            modulating_signal_SPL_from_voltage = modulating_signal_SPL_from_voltage-min(modulating_signal_SPL_from_voltage);
            modulating_signal_holder_converted = modulating_signal_SPL_from_voltage/max(modulating_signal_SPL_from_voltage);
            
            % Downsample modulating signal
            modulating_signal_holder_converted = modulating_signal_holder_converted';
            modulating_signal_holder_final = resample(modulating_signal_holder_converted,eeg_sampling_rate_downsampled_Hz,eeg_sampling_rate_original_Hz); %#ok<*SAGROW>
            clear modulating_signal_holder_converted
            
            % check data is the same sime
            stim_s = length(modulating_signal_holder_final);
            eeg_s = size(eeg_trial,1);
            adjust_data_length = min(stim_s,eeg_s);
            
            %% Store Data
            resp_holder{cross_validation_idx} = eeg_trial(1:adjust_data_length,1:128); clear eeg_trial
            stim_holder{cross_validation_idx} = modulating_signal_holder_final(1:adjust_data_length)';
            cross_validation_idx = cross_validation_idx+1;
        end
        
        lambda_test_values = [1e3,1e4];
        
        %% Decoding Analysis
        [rho,p_value,MSE,recon_eeg,stim_TRFmodel] = mTRFcrossval_Fix(stim,resp,eeg_sampling_rate_downsampled_Hz,mapping,...
            epoch_low_cutoff_SNR_ms,epoch_higher_cutoff_SNR_ms,lambda_test_values);
        
        % Select optimal lambda
        [best_labda_selected,best_lambda] = max(mean(rho,1));
        
        % stim_model [ trials by lambdas by chans by lags by feats) ]
        bmodel = squeeze(stim_TRFmodel(:,best_lambda,2:end));  % keep model for best lambda averaged across trials        
        stim_model_reshape = reshape(bmodel,[size(stim_TRFmodel,1),size(resp{1,1},2),size(bmodel,2)/size(resp{1,1},2)]);
                
        % >> Transform decoder weights to forward model
        lambda_value_plotting = lambda_test_values(best_lambda); % 1e3
        clear model_w model_tx
        for trl=1:length(trial_listings)
            
            % run mTRF train with same parameters as cross val model to get model for transforming
            [w_back,t_trn,con] = mTRFtrain_edit(stim{1,trl},resp{1,trl},fs,dir,epoch_low_cutoff_SNR_ms,epoch_higher_cutoff_SNR_ms,lambda_value_plotting);
            model_w(trl,:,:) = w_back;
            
            %% Transform model
            [modelt,t_tx] = mTRFtransform(stim{1,trl},resp{1,trl},w_back,fs,dir,epoch_low_cutoff_SNR_ms,epoch_higher_cutoff_SNR_ms,con);
            model_transfored(trl,:,:) = modelt(:,1:end);
        end
             
        % Avg transfored backward model
        trf = squeeze(mean(model_transfored));
        
        %% Save ReRef
        % Verify Directory Exists and if Not Create It
        if exist([data_path,'/',study_name,'/','Results','/',condition_name,'/',subjects{subject_idx},'/'],'dir') == 0
            mkdir([data_path,'/',study_name,'/','Results','/',condition_name,'/',subjects{subject_idx},'/']);
        end
        % Save Figures and Data
        filename = [data_path,'/',study_name,'/','Results','/',condition_name,'/',subjects{subject_idx},'/',...
            'mTRF_output']; filetype = '.mat';
        save([filename,filetype],'best_labda_selected','trf','recon_eeg','stim_model_reshape','-v7.3'); clear filename filetype
        clear eeg_trial
        
    end
end