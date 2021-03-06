function [RR,semSim,lme,amp_std,f0,resolve,on] = backwardsModel_KP2(stim,pred,sem,stim_e_f_r,Fs,tWind,onset)

% [RR,semSim,lme,amp_std,f0,resolve] = backwardsModel(stim,pred,sem,stim_e_f_r,c,Fs,tWind)
% Runs a seond stage regression with envelope prediction accuracy at the
% level of words as the dependent variable and various hierarchical
% features of speech as the predictors.
% INPUT:
% stim : The speech envelope. Cell matrix {1, nTrials}(Samples, 1)
%
% pred : Time series of the predicted speech envelope based on the backwards TRF. This can be computed with mTRFpredict/ mTRFtrain or mTRFcrossval. {1,nTrials}(Samples,
% nSubjects)
%
% sem : Semantic Similarity as an impulse vector. Note this must be
% perfectly alligned with the speech envelope. {1,nTrials)(Samples,1)
%
% stim_e_f_r: Prosody features. Columns 2 and 3 are f0 and resolvability
% {1,nTrials}(Sample,[envelope,f0,resolve])
%
% Fs: sampling rate (64Hz of 128Hz)
%
% tWind: Start of time window (in samples) of interest after onset of word. For example
% tWind = 0 (0-100ms), tWind = 6 (50-150ms)
%
% OUTPUT:
% RR: envelope reconstruction accuracies at a word level, (nWords,
% nSubjects)
%
% semSim: Semantic similarity values (nWords,1)
%
% lme: Linear mixed effects model
%
% amp_std: standard deviation of the envelope (nWords,1)
%
% f0: fundamental frequnecy (nWords,1)
%
% resolve: resolvability (nWords,1)
%
%
% Michael Broderick 2019


c=zeros(length(stim)+1,1);
windL=0.1; % 0.1 => 100 ms window length

for i=1:length(stim)
    
    sem{i}(end-round(windL*Fs):end)=0; % truncate semantic vectors
    c(i+1)=length(find(sem{i}));
    
end

% if Fs==64
%    for j=1:length(stim)
%     stim_e_f_r{j}=downsample(stim_e_f_r{j},2);
%    end
% end


semSim=zeros(sum(c),1);
amp_std=zeros(sum(c),1);
f0=zeros(sum(c),1);
resolve=zeros(sum(c),1);
on=zeros(sum(c),1);

RR=zeros(sum(c),size(pred{1},2),length(tWind));
count=1;
for wind=tWind
    fprintf('.');
    for s=1:size(RR,2)
        for runs=1:length(stim)
            f=round(find(sem{runs}));
            for i=1:length(f)-(ceil(windL*Fs)+wind)
            %for i=1:length(f)
                semSim(i+sum(c(1:runs)))=sem{runs}(f(i));
                
                amp_std(i+sum(c(1:runs)))=std(stim{runs}((f(i):f(i)+ceil(windL*Fs))+wind));
                f0(i+sum(c(1:runs)))=nanmean(stim_e_f_r{runs}((f(i):f(i)+ceil(windL*Fs))+wind,2));
                resolve(i+sum(c(1:runs)))=nanmean(stim_e_f_r{runs}((f(i):f(i)+ceil(windL*Fs))+wind,3));
                
                %on(i+sum(c(1:runs)))=onset{runs}(f(i));
                on(i+sum(c(1:runs)))= mean(onset{runs}((f(i):f(i)+ceil(windL*Fs))+wind,1));
                
                %RR(i+sum(c(1:runs)),s,count)=corr(stim{runs}((f(i):f(i)+ceil(windL*Fs))+wind),pred{runs}((f(i):f(i)+ceil(windL*Fs))+wind,s),'Type','Spearman');
                RR(i+sum(c(1:runs)),s,count)=corr(stim{runs}((f(i):f(i)+ceil(windL*Fs))+wind),pred{runs}((f(i):f(i)+ceil(windL*Fs))+wind,s),'Type','Pearson');
                
                
            end
        end
    end

        %semSim(1:end-1)=1-semSim(1:end);
        %semSim(1:end)=1-semSim(1:end);
        semSim=1-semSim;
        x1=zscore(semSim);  x1=repmat(x1,[size(RR,2),1]);
        x2=zscore(amp_std); x2=repmat(x2,[size(RR,2),1]);
             x3=zscore(f0); x3=repmat(x3,[size(RR,2),1]);
        x4=zscore(resolve); x4=repmat(x4,[size(RR,2),1]);
             x5=zscore(on); x5=repmat(x5,[size(RR,2),1]);
            
            X=[x1,x2,x3,x4,x1.*x2,x3.*x4];
             
            %X=[x1,x2,x3,x4,x5,x1.*x2,x3.*x4]; % >>>> use for SEM
                       
            %%X=[x1,x2,x1.*x2]; % >>> use for surprisal
            
            %X=[x1,x2,x3,x4,x5,x1.*x2,x3.*x4,x2.*x5];
            

        Y=RR(:,:,count);Y=Y(:);
        Z=ones(size(Y));
        u=ones(length(semSim),1)*(1:size(RR,2));u=u(:);
        lme=fitlmematrix(X,Y,Z,u);

    count=count+1;
end


% RR=zeros(sum(c),size(pred{1},2),length(tWind));
% count=1;
% for wind=tWind
%     fprintf('.');
%     for s=1:size(RR,2)
%         for runs=1:length(stim)
%             f=round(find(sem{runs}));
%             for i=1:length(f)
%                 semSim(i+sum(c(1:runs)))=sem{runs}(f(i));
%                 amp_std(i+sum(c(1:runs)))=std(stim{runs}((f(i):f(i)+ceil(windL*Fs))+wind));
%                 f0(i+sum(c(1:runs)))=mean(stim_e_f_r{runs}((f(i):f(i)+ceil(windL*Fs))+wind,2));
%                 resolve(i+sum(c(1:runs)))=mean(stim_e_f_r{runs}((f(i):f(i)+ceil(windL*Fs))+wind,3));
%                 RR(i+sum(c(1:runs)),s,count)=corr(stim{runs}((f(i):f(i)+ceil(windL*Fs))+wind),pred{runs}((f(i):f(i)+ceil(windL*Fs))+wind,s),'Type','Spearman');
%                 
%             end
%         end
%     end
%     
%     %semSim(1:end-1)=1-semSim(1:end);
%     semSim=1-semSim;
%     x1=zscore(semSim);x1=repmat(x1,[size(RR,2),1]);
%     x2=zscore(amp_std);x2=repmat(x2,[size(RR,2),1]);
%     x3=zscore(f0);x3=repmat(x3,[size(RR,2),1]);
%     x4=zscore(resolve);x4=repmat(x4,[size(RR,2),1]);
%     X=[x1,x2,x3,x4,x1.*x2,x3.*x4];
%     Y=RR(:,:,count);Y=Y(:);
%     Z=ones(size(Y));
%     u=ones(length(semSim),1)*(1:size(RR,2));u=u(:);
%     lme=fitlmematrix(X,Y,Z,u);
%     
%     count=count+1;
% end
