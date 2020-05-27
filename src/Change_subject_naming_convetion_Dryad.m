
% Summary:
% Script chnage naming convetion of Dryad data

% Status:
% Complete

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

%% Prepare variables for Cluster
% Manually Initialise Variables

%% Add subfolder/dir
data_path = 'E:\Semantic_Dissimilarity';
git_path = 'C:\Users\kevin\Documents\Github\Semantic_Dissimilarity_Project';

% Define study paths
study_path_E = 'E:\aa_Neuro_Typical_Database';
study_path = 'Z:\Neurotypical_Response_Inhibition';

% Define study folder
study_name = 'Cocktail_Party';

% Initialise Subject Variables
listing = dir([data_path,'/',study_name,'/','EEG_Data_orig','/']);
subejct_listings = {listing.name};
subejct_listings(cellfun('length',subejct_listings)<3) = [];
subejct_listings(end) = [];
subjects_orig = subejct_listings;
subjects_number = numel(subjects_orig);

subjects = cell(1,subjects_number);
str_1 = 'Subject_';
for k = 1:subjects_number
    if k < 10 ; str_1 = 'Subject_0'; else
        str_1 = 'Subject_'; end
    num_holder = num2str(k); subjects{k} = [str_1,num_holder];
end

% Correction for numerical sorting
subejct_listings = natsortfiles(subejct_listings);
subjects_orig = natsortfiles(subjects_orig);

%% Change naming convention to work with other projects
for subject_idx = 22:subjects_number
    subject_orig_name = subjects_orig{subject_idx};
    
    % Make new dir
    mkdir([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/']);
    newdir = ([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/']);
    % Copy subjct files from orig and chnage naming convention
    orig_dir  = ([data_path,'/',study_name,'/','EEG_Data_orig','/',num2str(subjects_orig{subject_idx}),'/','*.mat']);
    copyfile(orig_dir ,newdir)
    
    % List all files in subject subfolder   
    listing = dir([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/','*.mat']);
    run_listings = {listing.name};
    % Correction for numerical sorting
    run_listings = natsortfiles(run_listings);
    run_no = length(run_listings);
    disp([num2str(subject_idx),' - ',num2str(run_no)])
    
    % Rename all files in subject subfolder to match master folder
    for k = 1:length(run_listings)
        movefile([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/',subejct_listings{subject_idx},'_','Run',num2str(k),'.mat'],...
            [data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/',num2str(subjects{subject_idx}),'_','Run','_',num2str(k),'.mat']); %
    end
end

% % Make new dir
% mkdir([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/']);
% newdir = ([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/']);
% % Copy subjct files from orig and chnage naming convention
% orig_dir  = ([data_path,'/',study_name,'/','EEG_Data_orig','/',num2str(subjects_orig{subject_idx}),'/','*.mat']);
% copyfile(orig_dir ,newdir)
% % List all files in subject subfolder
% listing = dir([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/','*.mat']);
% run_listings = {listing.name};
% run_no = length(run_listings);
% disp([num2str(subject_idx),' - ',num2str(run_no)])
% 
% % Rename all files in subject subfolder to match master folder
% for k = 1:length(run_listings)
%     movefile([data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/',subejct_listings{k}],...
%         [data_path,'/',study_name,'/','EEG_Data','/',num2str(subjects{subject_idx}),'/',num2str(subjects{subject_idx})]); % ,'_','Run','_',num2str(k),'.mat'
% end
