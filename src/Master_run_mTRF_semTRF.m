
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
addpath  '/scratch/kprinslo/Semantic_Dissimilarity/Cocktail_Party/mTRF_KP_edit/';
addpath  '/scratch/kprinslo/Semantic_Dissimilarity/Cocktail_Party/Resources_Misc/';
addpath  '/scratch/kprinslo/Semantic_Dissimilarity/Cocktail_Party/Scripts/';
data_path = '/scratch/kprinslo/Semantic_Dissimilarity';
git_path = '/scratch/kprinslo/Semantic_Dissimilarity';
pc_path = '/scratch/kprinslo/Semantic_Dissimilarity';
fprintf('Added paths completed');

%% Add subfolder/dir
% addpath 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src'
% addpath 'C:\Users\kevin\Documents\Github\mTRF_KP_edit\'
% data_path = 'E:\Semantic_Dissimilarity';
% git_path = 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project';
% pc_path = 'C:\Users\kevin\Documents\Semantic_Dissimilarity';

% Define study folder
study_name = 'Cocktail_Party';

% pc_path = 'E:\Semantic_Dissimilarity';
% listing = dir(fullfile([data_path,'/',study_name,'/','Recordings_8Hz_kw','/']));

% Initialise Subject Variables
%e_path = 'E:/Semantic_Dissimilarity';
listing = dir(fullfile([pc_path,'/',study_name,'/','Recordings_30Hz','/']));
subejct_listings = {listing.name};
subejct_listings(cellfun('length',subejct_listings)<3) = [];
subjects_orig = subejct_listings;
subjects_number = numel(subjects_orig);
subjects = natsortfiles(subjects_orig); % Correction for numerical sorting
disp(subjects)

% Initialise EEG Variables
eeg_sampling_rate_original_Hz = 128;
Fs = eeg_sampling_rate_original_Hz;
channels_number_cephalic = 128;
eeg_sampling_rate_downsampled_Hz = 64;
eeg_trial_length = 60; % secs
eeg_trial_length_samples = eeg_sampling_rate_downsampled_Hz*eeg_trial_length;

% Define conditions
conditions = {'20000','Journey'};

% Define mTRF paramters
mapping = 1; % Backwards [-1] | Forwards [1]
lambda_test_values = 2.^(-8:2:32);

lambda_value_plotting = 1e3; % 1e2
multivariate_dimensions_number = 1; % define if multivariate (define n) or univariate (=1);
baseline_correction_TRF = 'BC'; % NA - for no BL TRF correction
epoch_low_cutoff_ms = -100;
epoch_higher_cutoff_ms = 800;

% Cluster parallel definition
subject_idx_cluster = str2double(getenv('SLURM_ARRAY_TASK_ID'));  % this give's you back the job parameter from slurm (#SBATCH --array=1-16)
disp(subject_idx_cluster)

%% mTRF Analysis

% subject_idx = 8
for subject_idx = subject_idx_cluster
    fprintf('Subject:\t\t\t\t\t\t%s\n',num2str(subject_idx));
    
    % subject_idx = 1;1:5 %1:length(subjects)
    for condition_idx = 1:length(conditions)
        condition_name = conditions{condition_idx};
        fprintf('Condition:\t\t\t\t\t\t%s\n',num2str(condition_idx));
        
        listing = dir(fullfile([pc_path,'/',study_name,'/','Recordings_30Hz','/',subjects{subject_idx},'/'],'*.mat'));
        trial_listings = {listing.name};
        trial_listings = natsortfiles(trial_listings); % Correction for numerical sorting
        
        fprintf('Begin Data loaded\n');
        % Load EEG Data
        stim = cell(1,length(trial_listings));
        stim_env = cell(1,length(trial_listings));
        resp = cell(1,length(trial_listings));
        cross_validation_idx = 1;
        for trial_idx = 1:length(trial_listings)
            fprintf('Trial:\t\t\t\t\t\t%s\n',num2str(trial_idx));
            
            % Deal with missing trials - index which stim to load that matches EEG
            str = trial_listings{trial_idx};
            str = str(12:end-4);
            stim_trial = string(regexp(str,'\d*','Match'));
            
            %% >> Load EEG data
            %pc_path_C = 'C:\Users\kevin\Documents\Semantic_Dissimilarity';
            load([pc_path,'/',study_name,'/','Recordings_30Hz','/',subjects{subject_idx},'/',...
                subjects{subject_idx},'_','Run',stim_trial{1},'.mat'],'eeg_trial');
            
            % Z-score EEG
            eeg_trial = eeg_trial';
            eeg_trial = zscore(eeg_trial(:,1:128));
            
            % Load modulating signal
            load([pc_path,'/',study_name,'/','Stimuli','/','Envelopes_Gammatone','/',condition_name,'/',...
                condition_name,'_',stim_trial{1},'_env.mat'],'envelope');
            
            % Load Semantic vectors
            load([pc_path,'/',study_name,'/','Stimuli','/','Semantic_vectors','/',condition_name,'_Glove_64Hz_Semantic_Similarity.mat'],'semVectors');
            
            % Define Word onsets
            clear onset sem semVectors_tmp
            on_tmp = semVectors{trial_idx};
            on_tmp(3840) = 0; % Buffer with zeros to match vector lengths
            on_tmp(on_tmp ~= 0) = 1;
            onset = on_tmp(1:3840);
            
            semVectors_tmp = semVectors{trial_idx};
            semVectors_tmp(3840) = 0; % Buffer with zeros to match vector lengths
            sem = semVectors_tmp(1:3840);            
            clear semVectors
                        
            envelope = envelope';
            modulating_signal_norm = envelope-min(envelope);
            modulating_signal_holder_converted = modulating_signal_norm/max(modulating_signal_norm);
            
            % check data is the same sime
            stim_s = length(modulating_signal_holder_converted);
            eeg_s = size(eeg_trial,1);
            adjust_data_length = min(stim_s,eeg_s);
            if adjust_data_length > eeg_trial_length_samples
                adjust_data_length = eeg_trial_length_samples;
            end
            
            %% Store Data
            resp{cross_validation_idx} = eeg_trial(1:adjust_data_length,1:128); clear eeg_trial
            stim{cross_validation_idx} = [sem(1:adjust_data_length)];
            %stim{cross_validation_idx} = [sem(1:adjust_data_length), onset(1:adjust_data_length)];
            %stim{cross_validation_idx} = [modulating_signal_holder_converted(1:adjust_data_length)];
            
            stim_env{cross_validation_idx} = [modulating_signal_holder_converted(1:adjust_data_length)];
            
            cross_validation_idx = cross_validation_idx+1;
        end
        fprintf('Data loaded\n');
        
        mapping = 1;
        
        %% Decoding Analysis
        [rho,p_value,MSE,pred_eeg,TRFmodel] = mTRFcrossval_Fix_final_KP(stim,resp,eeg_sampling_rate_downsampled_Hz,mapping,...
            epoch_low_cutoff_ms,epoch_higher_cutoff_ms,lambda_test_values);
        fprintf('Crossval completed');
        
        %% Select optimal lambda | [rho_max,sel] = max(mean(rho,1)) : 0.1106
        %channels_chosen1 = [61:63 54:56 106:108 115:117];
        %channels_chosen2 = [97 98 65 66 86 85];
        %channels_chosen = [channels_chosen1 channels_chosen2];
        
        channels_chosen = 1; %85
        %channels_chosen = 85; % 85 | 19
        
        rho_mean = squeeze(mean(rho,1));
        rho_mean_mean = squeeze(mean(rho_mean(:,channels_chosen),2));
        [rho_max, lambda_sel] = max(squeeze(mean(rho_mean(:,channels_chosen),2)));        
        
%         %% TRF
        % TRFmodel [30 11 121 128]  | size(TRFmodel)
        trf_mean = squeeze(mean(TRFmodel,1));
        trf_mean_chan = squeeze(mean(trf_mean(lambda_sel,2:end,channels_chosen),3));
        t = linspace(-0.1,0.8,59);
%         figure;plot(t,trf_mean_chan);
%                 
%         %% Plot Topo
%         channel_locations = readlocs([study_path_Y,'/','Resources_Misc','/','BioSemi','_',num2str(channels_number_cephalic),'_','AB','.sfp'],'filetype','sfp');
%         tm1 = round(0.445*64);
%         tm2 = round(0.5*64);
%         % tm1 = round(0.200*64);
%         % tm2 = round(0.250*64);
%         tmp = squeeze(trf_mean(lambda_sel,1:end,:));
%         tmp2 = squeeze(mean(tmp(tm1:tm2,:)));
%         figure;topoplot(tmp2,channel_locations(1,1:end),'Electrodes','Off','HeadRad',0.5);
        
        %tmp = trf_mean_chan(1:59);
        %tmp2 = trf_mean_chan(60:118);
        %tmp_mean = tmp+tmp2;
        %t = linspace(-0.1,0.8,59);
        %figure;plot(t,tmp_mean);
        
        %% Decoding Analysis
        [rho_env,p_value_env,MSE_env,pred_eeg_env,TRFmodel_env] = mTRFcrossval_Fix_final_KP(stim_env,resp,eeg_sampling_rate_downsampled_Hz,mapping,...
            epoch_low_cutoff_ms,epoch_higher_cutoff_ms,lambda_test_values);
        fprintf('Crossval completed');
                
        channels_chosen = 1; %85
                
        rho_mean_env = squeeze(mean(rho_env,1));
        rho_mean_mean_env = squeeze(mean(rho_mean_env(:,channels_chosen),2));
        [rho_max_env, lambda_sel_env] = max(squeeze(mean(rho_mean_env(:,channels_chosen),2)));        
        
%         %% TRF
        % TRFmode_env [30 11 121 128]  | size(TRFmodel_env)
        trf_mean_env = squeeze(mean(TRFmodel_env,1));
        trf_mean_chan_env = squeeze(mean(trf_mean_env(lambda_sel_env,2:end,channels_chosen),3));
        t_env = linspace(-0.1,0.8,59);
%         figure;plot(t_env,trf_mean_chan_env);
        
        %% Save ReRef
        % Verify Directory Exists and if Not Create It
        if exist([pc_path,'/',study_name,'/','Results_G_30Hz_sem','/',condition_name,'/',subjects{subject_idx},'/'],'dir') == 0
            mkdir([pc_path,'/',study_name,'/','Results_G_30Hz_sem','/',condition_name,'/',subjects{subject_idx},'/']);
        end
        % Save Figures and Data
        filename = [pc_path,'/',study_name,'/','Results_G_30Hz_sem','/',condition_name,'/',subjects{subject_idx},'/',...
            'mTRF_output']; filetype = '.mat';
          save([filename,filetype],'rho','stim','pred_eeg','TRFmodel','rho_env','stim_env','pred_eeg_env','TRFmodel_env','trf_mean_chan_env','trf_mean_chan','rho_max','rho_max_env','-v7.3'); clear filename filetype

%         save([filename,filetype],'rho','stim','best_labda_selected','best_lambda_hd','recon_stim','recon_stim_best_lambda','TRF_model_reshape','time_lags_fw','model_transfored','-v7.3'); clear filename filetype
        clear eeg_trial resp stim model_w recon_stim stim_model_reshape time_lags_fw model_transfored bmodel
        clear rho p_value MSE recon_stim TRFmodel TRF_model_reshape
        fprintf('Saving');
    end
end
