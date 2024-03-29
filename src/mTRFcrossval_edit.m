function [r,p,mse,pred,model] = mTRFcrossval_edit(stim,resp,fs,map,tmin,tmax,lambda)
%mTRFcrossval mTRF Toolbox cross-validation function.
%   [R,P,MSE] = MTRFCROSSVAL(STIM,RESP,FS,MAP,TMIN,TMAX,LAMBDA) performs
%   leave-one-out cross-validation on the set of stimuli STIM and the
%   neural responses RESP for the range of ridge parameter values LAMBDA.
%   As a measure of performance, it returns the correlation coefficients R
%   between the predicted and original signals, the corresponding p-values
%   P and the mean squared errors MSE. Pass in MAP==1 to map in the forward
%   direction or MAP==-1 to map backwards. The sampling frequency FS should
%   be defined in Hertz and the time lags should be set in milliseconds
%   between TMIN and TMAX.
%
%   [...,PRED,MODEL] = MTRFCROSSVAL(...) also returns the predictions PRED
%   and the linear mapping functions MODEL.
%
%   Inputs:
%   stim   - set of stimuli [cell{1,trials}(time by features)]
%   resp   - set of neural responses [cell{1,trials}(time by channels)]
%   fs     - sampling frequency (Hz)
%   map    - mapping direction (forward==1, backward==-1)
%   tmin   - minimum time lag (ms)
%   tmax   - maximum time lag (ms)
%   lambda - ridge parameter values
%
%   Outputs:
%   r      - correlation coefficients
%   p      - p-values of the correlations
%   mse    - mean squared errors
%   pred   - prediction [MAP==1: cell{1,trials}(lambdas by time by chans),
%            MAP==-1: cell{1,trials}(lambdas by time by feats)]
%   model  - linear mapping function (MAP==1: trials by lambdas by feats by
%            lags by chans, MAP==-1: trials by lambdas by chans by lags by
%            feats)
%
%   See README for examples of use.
%
%   See also LAGGEN MTRFTRAIN MTRFPREDICT MTRFMULTICROSSVAL.

%   References:
%      [1] Crosse MC, Di Liberto GM, Bednar A, Edmund EC (2015). The
%          multivariate temporal response function (mTRF) toolbox: a MATLAB
%          toolbox for relating neural signals to continuous stimuli.
%          Frontiers in Human Neuroscience, doi: 10.3389/fnhum.2016.00604.

%   Author: Michael Crosse
%   Lalor Lab, Trinity College Dublin, IRELAND
%   Email: edmundlalor@gmail.com
%   Website: http://lalorlab.net/
%   April 2014; Last revision: 31 May 2016

% Define x and y
if tmin > tmax
    error('Value of TMIN must be < TMAX')
end
if map == 1
    x = stim;
    y = resp;
elseif map == -1
    x = resp;
    y = stim;
    [tmin,tmax] = deal(tmax,tmin);
else
    error('Value of MAP must be 1 (forward) or -1 (backward)')
end
clear stim resp

% Convert time lags to samples
tmin = floor(tmin/1e3*fs*map);
tmax = floor(tmax/1e3*fs*map);
% changed this for single lag decoder
% tmax = ceil(tmax/1e3*fs*map);

% Set up regularisation
% for reconstruction identity matrix is always used because EEG is x (i.e., is multivariate)
% dim1 = size(x{1},2)*length(tmin:tmax)+size(x{1},2); %128*nLags + 128
dim1 = size(x{1},2)*length(tmin:tmax)+1; %nFeats*nLags + 1 (constant term)
dim2 = size(y{1},2);
model = zeros(numel(x),numel(lambda),dim1,dim2);
% if size(x{1},2) == 1
%     d = 2*eye(dim1,dim1); d([1,end]) = 1;
%     u = [zeros(dim1,1),eye(dim1,dim1-1)];
%     l = [zeros(1,dim1);eye(dim1-1,dim1)];
%     M = d-u-l;
% else
% need to use identity matrix for regularization if comparing with multi-d models so that comparison is fair
    M = eye(dim1,dim1);
% end

%***** NB - need to exclude the constant term from regularisation
% - see Hastie, Friedman and Tibrishani book
% - this means setting the first n columns of the identity matrix to zero
% - for forward modelling n is the no. of features in the stimulus
% - for backward modelling n is the no. of EEG channels

%     M(1:size(x{1},2),:) = 0;
     M(1,:) = 0; % don't regularize constant term

% Training
X = cell(1,numel(x));
for i = 1:numel(x)
    % Generate lag matrix
%     X{i} = [ones(size(x{i})),lagGen(x{i},tmin:tmax)];
    X{i} = [ones(size(x{i},1),1),lagGen(x{i},tmin:tmax)];
    % Calculate model for each lambda value
    for j = 1:length(lambda)
        model(i,j,:,:) = (X{i}'*X{i}+lambda(j)*M)\(X{i}'*y{i}); % see end note
    end
end

% Testing
pred = cell(1,numel(x));
r = zeros(numel(x),numel(lambda),dim2);
p = zeros(numel(x),numel(lambda),dim2);
mse = zeros(numel(x),numel(lambda),dim2);
for i = 1:numel(x)
    pred{i} = zeros(numel(lambda),size(y{i},1),dim2);
    % Define training trials
    trials = 1:numel(x);
    trials(i) = [];
    % Perform cross-validation for each lambda value
    for j = 1:numel(lambda)
        % Calculate prediction
        pred{i}(j,:,:) = X{i}*squeeze(mean(model(trials,j,:,:),1)); 
        % Calculate accuracy
        for k = 1:dim2
            [r(i,j,k),p(i,j,k)] = corr(y{i}(:,k),squeeze(pred{i}(j,:,k))');
            mse(i,j,k) = mean((y{i}(:,k)-squeeze(pred{i}(j,:,k))').^2);
        end
    end
end

end

%%
% The equation for getting the model is the solution to the minimisation of ||Ax-b||^2 + ||Gx||^2,
% where A is the input, x is the model (mapping/transformation), b is the
% output and G = lambda*I is the regularisation. 
% This is a smoothness term which helps avoid overfitting
% Anytime you need to solve an inverse problem it may be ill-posed
% The inverse is not ill-posed if a solution exists, is unique and is stable. 