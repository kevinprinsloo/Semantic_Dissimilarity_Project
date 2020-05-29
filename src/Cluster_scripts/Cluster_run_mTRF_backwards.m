
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

%% Add subfolder/dir
%addpath 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project\src'
%addpath 'C:\Users\kevin\Documents\Github\mTRF_KP_edit\'
%data_path = 'E:\Semantic_Dissimilarity';
%git_path = 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project';
%pc_path = 'C:\Users\kevin\Documents\Semantic_Dissimilarity';

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

% Cluster parallel definition
subject_idx_cluster = str2double(getenv('SLURM_ARRAY_TASK_ID'));  % this give's you back the job parameter from slurm (#SBATCH --array=1-16)
disp(subject_idx_cluster)

%% mTRF Analysis
for subject_idx = subject_idx_cluster
    % subject_idx = 1;1:5 %1:length(subjects)
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
