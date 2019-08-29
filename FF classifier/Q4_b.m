clear; clc

fid=fopen('4.1.txt');
label=textscan(fid,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s','whitespace',',');
fclose(fid);

Feats = str2double([label{2}, label{3}, label{4}, label{5}, label{6}, label{7}, ...
    label{8}, label{9}, label{10}, label{11}, label{12}, label{13}, ...
    label{14}]);

Tar = str2double(label{1});
nnTar = mat2label(Tar');

mT = mean(Feats);
mSd = std(Feats);

NormFeats = bsxfun(@minus, Feats, mT);
NormFeats = bsxfun(@rdivide, NormFeats, mSd)';

%3 we figured out the best performance was given when hidden nodes are : 

net = newff(NormFeats, nnTar, [3 7 9]);
net.divideParam.trainRatio = 0.75;
net.divideParam.testRatio = 0.25;
net.divideParam.valRatio = 0.0;
net.trainParam.epochs = 100;

net =train(net, NormFeats, nnTar);

Q4_Test; %it outputs [a b c] that are used to test the data

% Normalizing the data, here is a very importnt point from the concept
% point of view, it's that you must normalize the data using the mean and
% the standard deviation extracted from the TRAINING set 
NormTest = bsxfun(@minus, [a, b, c], mT');
NormTest = bsxfun(@rdivide, NormTest, mSd');

netOut = sim(net, NormTest);
[~, Class] = max(netOut)

