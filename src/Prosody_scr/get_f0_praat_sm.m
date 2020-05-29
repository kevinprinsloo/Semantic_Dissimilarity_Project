function [f0, t] = get_f0_praat_sm(y, sr, eeg_fs)

f0_sr = eeg_fs; % default = 100 Hz
f0_lowerbound = 50; % default = 75
f0_upperbound = 500; % default 500
f0_maxcandidates = 15; % default = 15
f0_veryaccurate = 'no'; % default = 'no'
f0_silencethresh = 0.03; % default = 0.03;
f0_voicethresh = 0.45; % default = 0.45;
f0_octavecost = 0.01; % default = 0.01;
f0_octavejump = 0.35; % default = 0.35;
f0_voicetrans = 0.14; % default = 0.14;

praat_directory = strrep(which('get_f0_praat.m'),'get_f0_praat.m','');

% write temporary wav file
y = 0.005 * y / sqrt(mean(y(:).^2));
inputfile = [praat_directory 'tmp' DataHash(y) '.wav'];
audiowrite(inputfile,y,sr);

% calculate f0s and write to temporary outputfile
outputfile = [praat_directory 'tmp' DataHash(y) '.txt'];
if exist(outputfile,'file')
    delete(outputfile); 
end

call_praat = [praat_directory 'Praat.exe ' '--run ' praat_directory 'praat_f0.praat'];
system([call_praat ' '  inputfile  ' '   outputfile ' ' num2str(f0_lowerbound) ' ' num2str(f0_upperbound) ' ' num2str(1/f0_sr)  ' ' ...
    num2str(f0_maxcandidates) ' ' f0_veryaccurate ' ' num2str(f0_silencethresh) ' ' num2str(f0_voicethresh) ' '...
    num2str(f0_octavecost) ' ' num2str(f0_octavejump) ' ' num2str(f0_voicetrans)]);   

% read file with f0 values
fid = fopen(outputfile,'r');
x = textscan(fid,'%f%f'); fclose(fid);
t = x{1};
f0 = x{2};

% delete temporary files
delete(inputfile);
delete(outputfile);