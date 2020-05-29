function [f_rel, res] = measure_prosody_sm(file_name, file_directory, fs) 
% measures a collection of acoustic statistics from a wav file ('file_name.wav')
% statistics are returned in a structure S, and parameters in a structure P
% by default, a number of plots are generated showing the measures computed
% the statistics and figures are saved, and the directory to which they are saved can be specified by the 4th argument

% fs is the sampling frequency of the output measures
% modified by Emily Teoh (2019) based on Norman-Haignere's measure_stats.m script

file_name = strrep(file_name,'.wav','');
%file_name = strrep(file_name,''); % KP edit

parent_directory = strrep(which('measure_prosody.m'),'measure_prosody.m','');

% plotting
if file_directory(end) ~= '/'
    file_directory = [file_directory,'/'];
end

if ~contains(path, 'AuditoryToolbox')
    addpath(genpath([parent_directory 'AuditoryToolbox']));
end

if ~contains(path, 'Sound_Texture_Synthesis_Toolbox')
    addpath(genpath([parent_directory 'Sound_Texture_Synthesis_Toolbox']));
end

if ~contains(path, 'nsltools_texture_synthesis')
    addpath(genpath([parent_directory 'nsltools_texture_synthesis']));
end

% parameters for Josh's texture synthesis
synthesis_parameters; % loads default parameters
P.pad_duration = 2;
P.audio_sr = 20000;
P.hi_audio_f = P.audio_sr/2;
P.max_orig_dur_s = inf;
P.N_audio_channels = round(30 * log2(P.hi_audio_f/P.low_audio_f)/log2(10000/20)); % number of filters chosen to mimic gammatone filter bandwidths
P.use_more_audio_filters = 1; % 2x overcomplete
P.measurement_windowing = 1; % no window
P.orig_sound_filename = [file_name '.wav'];
P.orig_sound_folder = file_directory;

% spectrotemporal parameters for Shamma model
P.lowpass = [1 1];
P.highpass = [0 0];
P.dc_component = 0;
P.paras = [1000/P.env_sr, NaN, NaN, NaN]; % frame time in ms, other three parameters related to custom auditory model
P.rv = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]; % temporal rates in Hz, logspace(log10(P.low_mod_f), log10(P.hi_mod_f), P.N_mod_channels); % [0.5, 1, 2, 4, 8, 16, 32, 64, 128]; % temporal rates
P.sv = [0.125, 0.25, 0.5, 1, 2, 4, 8]; % spectral rates in cyc / octave

%% Default stats from McDermott texture model
y = format_orig_sound_without_rmsnorm(P);
measurement_win = set_measurement_window(length(y),P.measurement_windowing,P);
if P.pad_duration > 0
    y_nopad = y;
    y = [y_nopad; zeros(P.audio_sr*P.pad_duration,1)];
    measurement_win = [measurement_win; zeros(P.env_sr*P.pad_duration,1)];
end
[S, subband_envs] = measure_texture_stats(y,P,measurement_win);
S = edit_measured_stats(S,P);

% subband time vector
S.t = (0:size(subband_envs,1)-1)'/P.env_sr;
S.dur = S.t(end) - P.pad_duration; % duration without padding
subband_envs_nopad = subband_envs(S.t <= S.dur,:);

%% Resolvability
trunc_thresh = 0.08 * rms(y_nopad)/0.025;
px_laplace = peaks_laplace_specgram(subband_envs_nopad', S.f, 2, trunc_thresh, 1/P.env_sr, 'false', 'loginput');
px_laplace_trunc = max(px_laplace,0);
S.resolvability = max(px_laplace_trunc,[],1)';
res = resample(S.resolvability,fs,400);

%% F_rel
y_nopad = y(1:round(P.audio_sr*S.dur));
[S.f0, S.f0_t] = get_f0_praat(y_nopad, P.audio_sr);
f0 = S.f0;
f_rel = (f0 - nanmean(f0)) ./ std(f0(~isnan(f0))); % z score normalized
f_rel(isnan(f_rel)) = 0;