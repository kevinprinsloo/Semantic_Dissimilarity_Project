function [semVec,onsetVec,words,start] = createTimeVectors(filename,words,semVal,Fs,LEN)

if strcmp(filename(end-3:end),'Grid')
    
    [labels, a, b] = func_readTextgrid(filename);
    wordVec=labels{2};
    start=a{2};
    stop=b{2};
    
else
    fid=fopen(filename);
    headings=textscan(fid,'%s',3,'Delimiter',{','});clear headings
    words_pos=textscan(fid,'%s %f %f','Delimiter',',');
    fclose(fid);
    wordVec=words_pos{1};
    start=words_pos{2};
    stop=words_pos{3};
end

for i=1:length(wordVec)
   try if strcmp(lower(wordVec{i}(end-1:end)),"'s")
           wordVec{i}(end-1:end)=[];
       else
           if strcmp(lower(wordVec{i}(end-1:end)),"s'")
           wordVec{i}(end)=[];

           end
       end
   catch
   end

    
end

start(~ismember(lower(wordVec),words),:)=[];
stop(~ismember(lower(wordVec),words),:)=[];
wordVec(~ismember(lower(wordVec),words),:)=[];



if ~exist('LEN','var') || isempty(LEN)
    semVec=zeros(round(stop(end)*Fs)+20,size(semVal,2));
    onsetVec=zeros(round(stop(end)*Fs)+20,1);
else
    semVec=zeros(LEN*Fs,size(semVal,2));
    onsetVec=zeros(LEN*Fs,1);
end

if size(semVal,2)>1
    semVal(1,:)=[];
else
    semVal(1)=[];
end
start(1)=[];stop(1)=[];wordVec(1)=[];words(1)=[];
semVec(round(start*Fs),:)=semVal';
onsetVec(round(start*Fs),1)=mean(semVal);

end

