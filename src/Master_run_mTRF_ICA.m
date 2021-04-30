
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
addpath 'C:\Users\kevin\Documents\Github\Filtering';

% Manually Initialise Variables

%% Add subfolder/dir
data_path = 'E:\Semantic_Dissimilarity';
git_path = 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project';
pc_path = 'C:\Users\kevin\Documents\Semantic_Dissimilarity';

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
epoch_higher_cutoff_SNR_ms = 600;
epoch_low_cutoff_SNR_ms = 0;

%FILTER
FstopH = 0.1;
FpassH = 1;
AstopH = 65;
FpassL = 8;
FstopL = 9;
AstopL = 65;
Apass = 1;

fs = 64;
% Generate high/low-pass filters
h = fdesign.highpass(FstopH,FpassH,AstopH,Apass,fs);
hpf = design(h,'cheby2','MatchExactly','stopband'); clear h
h = fdesign.lowpass(FpassL,FstopL,Apass,AstopL,fs);
lpf = design(h,'cheby2','MatchExactly','stopband'); clear h
% fvtool(hpf,lpf)

%% Preprocess data
for subject_idx = 32 %1:length(subjects)
    
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
        
        % Apply filter - column will be filtered
        flteeg = filtfilthd(hpf,eegData); clear eegData
        flteeg = filtfilthd(lpf,flteeg);
        
        %% Downsample Data
        for chanIdx = 1:size(flteeg,2)
            eeg_temp(:,chanIdx) = resample(flteeg(:,chanIdx),eeg_sampling_rate_downsampled_Hz,eeg_sampling_rate_original_Hz); %#ok<*SAGROW>
        end
        eeg_trial = eeg_temp; clear eeg_temp
        
        % Concatinate all epochs
        eeg_holder(trial_idx,:,:) = eeg_trial(1:3840,:); clear eeg_trial
    end
    
    %% Remove bad channels
    % Load channel montage
    nChans = 128;
    channels_number_cephalic = size(eeg_holder,3);
    channel_locations = readlocs([git_path,'/','Resources_Misc','/','BioSemi','_',num2str(channels_number_cephalic),'_','AB','.sfp'],'filetype','sfp');
    channel_labels = cell(1,128);
    for k = 1:128
        channel_labels(k) = cellstr(channel_locations(k).labels);
    end
    
    % Put data into EEGLab data structure
    eeg_holder = permute(eeg_holder,[3 2 1]); % prepare for EEGLab structure
    
    % Insert data into EEGLab structure
    clear EEG
    EEG.data = eeg_holder; EEG.nbchan = channels_number_cephalic;
    EEG.srate = 64;
    EEG.chanlocs = channel_locations;
    EEG.trials = size(eeg_holder,3);
    EEG.pnts = size(eeg_holder,2);
    EEG.xmin = 1; EEG.xmax = round((size(eeg_holder,2)/64)*1000);
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
    
    % Select bad channel removal thresholds
    [~,idx1,~,~] = pop_rejchan(EEG,'elec',1:nChans,'threshold',3,...
        'norm','on','measure','kurt');
    [~,idx2,~,~] = pop_rejchan(EEG,'elec',1:nChans,'threshold',3,...
        'norm','on','measure','prob');
    [~,idx3,~,~] = pop_rejchan(EEG,'elec',1:nChans,'threshold',3,...
        'norm','on','measure','spec');
    
    idx1 = reshape(idx1,1,[]);
    idx2 = reshape(idx2,1,[]);
    idx3 = reshape(idx3,1,[]);
    badChans = unique([idx1,idx2,idx3]);
    
    % Spline interpolate bad channels
    if ~isempty(badChans)
        EEG = pop_interp(EEG,badChans,'spherical');
    end
    eeg_data = double(EEG.data);
    
    %% Re-reference to common average
    eeg_trial_clean = zeros(size(eeg_data));
    for k = 1:size(eeg_data,3)
        eeg_trial = eeg_data(:,:,k);
        dat = ft_preproc_rereference(eeg_trial, 'all', 'avg'); clear eeg_trial
        %dat = ft_preproc_rereference(eeg_trial, [129 130], 'avg'); clear eeg_trial
        eeg_trial_clean(:,:,k) = dat; clear dat
    end
    clear eeg_data
    
    %     %---------
    %     %% ICA
    %     %---------
    %
    %     % Apply ICA
    %     clear C A
    %     A = eeg_trial_clean; clear dat_temp
    %     C = reshape(A,size(A,1),[]);
    %
    %     %% Test reshape
    %     %figure;plot(eeg_trial_clean(1,1:100,2)); axis tight; hold on
    %     %plot(C(1,3842:3942));
    %
    %     % Put data into EEGLab data structure
    %     clear EEG
    %     nChans = channels_number_cephalic;
    %     EEG.data = C; EEG.nbchan = nChans;
    %     EEG.srate = 64;
    %     EEG.chanlocs = channel_locations;
    %     EEG.trials = 1;
    %     EEG.pnts = size(C,2);
    %     EEG.xmin = [];
    %     EEG.xmax = [];
    %     EEG.icaact = [];
    %     EEG.epoch = [];
    %     EEG.setname = 'dataX_epoch';
    %     EEG.filename = ''; EEG.filepath = '';
    %     EEG.subject = ''; EEG.group = '';
    %     EEG.condition = ''; EEG.session = [];
    %     EEG.comments = 'preprocessed with fieldtrip';
    %     EEG.times = [];
    %     EEG.event = [];
    %     EEG.icawinv = []; EEG.icasphere = [];
    %     EEG.icaweights = []; EEG.icaact = [];
    %     EEG.saved = 'no'; EEG.etc = [];
    %     EEG.stats = [];
    %     EEG.etc = [];
    %     EEG.specdata = [];
    %     EEG.specicaact = [];
    %     EEG.icaact = [];
    %     EEG.icawinv = [];
    %     EEG.icasphere  = [];
    %     EEG.icaweights = [];
    %     EEG.icachansind = [];
    %     EEG.ref = 'averef';
    %
    %     dataRank = 128;
    %     [icaweights,icasphere] = runica(EEG.data,'pca',dataRank,'extended',1);
    %
    %     icawinv = pinv(icaweights*icasphere);
    %     icachansind = 1:nChans;
    %     icaact = icaweights*icasphere*C;
    %
    %     % ICA Weight transfering
    %     EEG.icaweights = icaweights;
    %     EEG.icasphere = icasphere;
    %     EEG.icawinv = icawinv;
    %     EEG.icachansind = icachansind;
    %     EEG.icaact = icaact;
    %
    %     % ICLabel
    %     EEG = iclabel(EEG);
    %     EEG.data = double(EEG.data);
    %     EEG.icaact = double(EEG.icaact);
    %
    %     eyeNDX = find(strcmpi(EEG.etc.ic_classification.ICLabel.classes,'Eye'));
    %     [~,Comps] = max(EEG.etc.ic_classification.ICLabel.classifications>0.75,[],2);
    %     eyeComps = find(Comps == eyeNDX);
    %     noiseNDX = find(contains(EEG.etc.ic_classification.ICLabel.classes,'Noise'));
    %     noiseComps = find(any(Comps == noiseNDX,2));
    %     muscNDX = find(contains(EEG.etc.ic_classification.ICLabel.classes,'Muscle'));
    %     muscComps = find(Comps == muscNDX);
    %     Comps2Rem = [eyeComps(:); noiseComps(:); muscComps(:)];
    %
    %     % remove the eye and noise sources
    %     if ~isempty(Comps2Rem)
    %         % use the EEG lab function to remove the eye components
    %         EEG = pop_subcomp(EEG, Comps2Rem);
    %     end
    %
    %     % Keep only candidate brain components: (prob_brain + prob_other)
    %     % >= 50%
    %     brainNDX = find(contains(EEG.etc.ic_classification.ICLabel.classes,'Brain'));
    %     otherNDX = find(contains(EEG.etc.ic_classification.ICLabel.classes,'Other'));
    %     probBrainOther = EEG.etc.ic_classification.ICLabel.classifications(:,brainNDX) + EEG.etc.ic_classification.ICLabel.classifications(:,otherNDX);
    %     comps2Keep = find(probBrainOther>= 0.5);
    %
    %     if ~isempty(comps2Keep)
    %         % use the EEG lab function to remove the eye components
    %         EEG = pop_subcomp(EEG, comps2Keep,0,1);
    %     end
    %
    %     tmp = EEG.data;
    %     clear tmp_dat
    %     tmp_dat = reshape(tmp,[nChans,size(eeg_trial_clean,2), size(eeg_trial_clean,3)]);
    
    % Create EEG FT structure
    eeg_matrix = cell(1,length(trial_listings));
    for trlidx = 1:length(trial_listings)
        eeg_matrix{trlidx} = squeeze(eeg_trial_clean(:,:,trlidx));
    end
    
    % Craete Time/Samples FT structure
    trial_length_samples = round(60*64);
    t_stimulus = cell(1,length(trial_listings));
    for trlidx = 1:length(trial_listings)
        t_stimulus{trlidx} =  linspace(0,1,1*trial_length_samples);
    end
    
    % Create Fieldtrip data structure
    clear data
    data.label      = channel_labels';
    data.fsample    = 64;
    data.trial      = eeg_matrix;
    data.time       = t_stimulus;
    data.dimord     = 'rpt_chan_time';
    data.sampleinfo = [1 size(t_stimulus{trlidx},2)];
    
    cfg = [];
    cfg.continuous = 'yes';
    cfg.detrend = 'no';
    cfg.demean = 'no';
    data = ft_preprocessing(cfg,data);
        
    %% Independent component analysis (ICA) to remove eye blinks and movements
    cfg = [];
    cfg.method = 'runica';
    cfg.numcomponent = 128;
    cfg.demean = 'yes';
    cfg.channel = {'EEG'};
    ica = ft_componentanalysis(cfg, data);
    
    % plot components
    cfg = [];
    cfg.component = 1:30;
    cfg.layout = ['biosemi',num2str(channels_number_cephalic),'.lay'];
    cfg.comment = 'no';
    ft_topoplotIC(cfg, ica);
    cfg = [];
    cfg.layout = ['biosemi',num2str(channels_number_cephalic),'.lay'];
    cfg.viewmode = 'component';
    ft_databrowser(cfg, ica);
    
    % remove appropriate components
    cfg = [];
    cfg.component = [1 3 4 5 11 15 21 22 23 26 30]; % NB! these should only be the eye related components
    cfg.demean = 'no';
    ica_removed = ft_rejectcomponent(cfg, ica, data);
    close all
        
    dat_cl = ica_removed.trial; clear ica_removed  
    for trial_idx = 1:length(trial_listings)
        
        % Extract epoch
        eeg_trial = dat_cl{trial_idx};
        
        %% Save ReRef
        % Verify Directory Exists and if Not Create It
        if exist([data_path,'/',study_name,'/','Recordings_8Hz_ica','/',subjects{subject_idx},'/'],'dir') == 0
            mkdir([data_path,'/',study_name,'/','Recordings_8Hz_ica','/',subjects{subject_idx},'/']);
        end
        % Save Figures and Data
        filename = [data_path,'/',study_name,'/','Recordings_8Hz_ica','/',subjects{subject_idx},'/',...
            trial_listings{trial_idx}];
        save(filename,'eeg_trial','-v7.3'); clear filename filetype
        clear eeg_trial
    end
end

%% mTRF Analysis
for subject_idx = 1:5 %1:length(subjects)
    for condition_idx = 1:length(conditions)
        condition_name = conditions{condition_idx};
        
        listing = dir([pc_path,'/',study_name,'/','Recordings','/',num2str(subjects{subject_idx}),'/','*.mat']);
        trial_listings = {listing.name};
        trial_listings = natsortfiles(trial_listings); % Correction for numerical sorting
        
        listing = dir([pc_path,'/',study_name,'/','Stimuli','/','Envelopes','/',condition_name,'/','*.mat']);
        stim_listings = {listing.name};
        stim_listings = natsortfiles(stim_listings); % Correction for numerical sorting
        
        % Load EEG Data
        stim = cell(1,length(trial_listings));
        resp = cell(1,length(trial_listings));
        cross_validation_idx = 1;
        for trial_idx = 1:length(trial_listings)
            
            %% >> Load EEG data
            load([pc_path,'/',study_name,'/','Recordings','/',subjects{subject_idx},'/',...
                trial_listings{trial_idx}],'eeg_trial');
            
            % Z-score EEG
            eeg_trial = eeg_trial';
            eeg_trial = zscore(eeg_trial(:,1:128));
            
            % Deal with missing trials - index which stim to load that matches EEG
            str = trial_listings{trial_idx};
            str = str(9:end-4); stim_trial = string(regexp(str,'\d*','Match'));
            
            % Load modulating signal
            load([pc_path,'/',study_name,'/','Stimuli','/','Envelopes','/',condition_name,'/',...
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
            resp{cross_validation_idx} = eeg_trial(1:adjust_data_length,1:128); clear eeg_trial
            stim{cross_validation_idx} = modulating_signal_holder_final(1:adjust_data_length)';
            cross_validation_idx = cross_validation_idx+1;
        end
        
        %% Decoding Analysis
        [rho,p_value,MSE,recon_eeg,stim_TRFmodel] = mTRFcrossval_Fix_final_KP(stim,resp,eeg_sampling_rate_downsampled_Hz,mapping,...
            epoch_low_cutoff_SNR_ms,epoch_higher_cutoff_SNR_ms,lambda_test_values);
        
        % Select optimal lambda
        [best_labda_selected,best_lambda] = max(mean(rho,1));
        
        % stim_model [ trials by lambdas by chans by lags by feats) ]
        bmodel = squeeze(stim_TRFmodel(:,best_lambda,2:end));  % keep model for best lambda averaged across trials
        stim_model_reshape = reshape(bmodel,[size(stim_TRFmodel,1),size(resp{1,1},2),size(bmodel,2)/size(resp{1,1},2)]);
        
        %stim_TRFmodel2 = stim_TRFmodel(:,:,2:end);
        %trf_plot = reshape(stim_TRFmodel2,[size(stim_TRFmodel2,1),size(stim_TRFmodel2,2),size(resp{1,1},2),size(stim_TRFmodel2,3)/size(resp{1,1},2)]);
        %size(trf_plot)
        
        % >> Transform decoder weights to forward model
        lambda_value_plotting = lambda_test_values(best_lambda); % 1e3
        clear model_w model_tx model_transfored
        for trl=1:length(trial_listings)
            
            % run mTRF train with same parameters as cross val model to get model for transforming
            [w_back,t_trn,con] = mTRFtrain_Fix_KP(stim{1,trl},resp{1,trl},eeg_sampling_rate_downsampled_Hz,...
                mapping,epoch_low_cutoff_SNR_ms,epoch_higher_cutoff_SNR_ms,lambda_value_plotting);
            model_w(trl,:,:) = w_back;
            
            %% Transform model
            [modelt,t_tx] = mTRFtransformModified_Final_KP(stim{1,trl},resp{1,trl},w_back,eeg_sampling_rate_downsampled_Hz,...
                mapping,epoch_low_cutoff_SNR_ms,epoch_higher_cutoff_SNR_ms,con);
            model_transfored(trl,:,:) = modelt;
        end
        time_lags_fw = fliplr(t_trn); % flip time lags used
        
        %% Save ReRef
        % Verify Directory Exists and if Not Create It
        if exist([pc_path,'/',study_name,'/','Results','/',condition_name,'/',subjects{subject_idx},'/'],'dir') == 0
            mkdir([pc_path,'/',study_name,'/','Results','/',condition_name,'/',subjects{subject_idx},'/']);
        end
        % Save Figures and Data
        filename = [pc_path,'/',study_name,'/','Results','/',condition_name,'/',subjects{subject_idx},'/',...
            'mTRF_output']; filetype = '.mat';
        save([filename,filetype],'best_labda_selected','recon_eeg','stim_model_reshape','time_lags_fw','model_transfored','-v7.3'); clear filename filetype
        clear eeg_trial resp stim model_w recon_eeg stim_model_reshape time_lags_fw model_transfored bmodel
        clear rho p_value MSE recon_eeg stim_TRFmodel
    end
end
