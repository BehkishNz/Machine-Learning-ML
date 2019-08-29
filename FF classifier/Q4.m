clear; clc
format long
% So we start with loading the data from the submitted text file 
fid=fopen('4.1.txt');
label=textscan(fid,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s','whitespace',',');
fclose(fid);
% Then we extract the data from it, the features are in the second to the
% last column. note that it must converted from string because of the text
Feats = str2double([label{2}, label{3}, label{4}, label{5}, label{6}, label{7}, ...
    label{8}, label{9}, label{10}, label{11}, label{12}, label{13}, ...
    label{14}]);
% Here is the target taking the first column of the data
Tar = str2double(label{1});

% It takes the original target [1 2 3] and convert that to a format which a single
% target is active at a time which is required to train the net. 
nnTar = mat2label(Tar');

% Here taking the mean and the standerd deviation from the training data.
mT = mean(Feats);
mSd = std(Feats);

% here we do actual normalization with each feature being subtracted from
% it's mean and divided from it's standard deviation. this is called the
% broadcasting and it's done by the bsxfun command
NormFeats = bsxfun(@minus, Feats, mT);
NormFeats = bsxfun(@rdivide, NormFeats, mSd)';

MinError = 100;
MinI = 0;
MinJ = 0;
Minz = 0;
% 1 hidden layer
for i = 1: 10
            net = newff(NormFeats, nnTar, i);
            net.divideParam.trainRatio = 0.75;
            net.divideParam.testRatio = 0.25;
            net.divideParam.valRatio = 0.0;
            net.trainParam.epochs = 100;

            net =train(net, NormFeats, nnTar);
            Y = net(NormFeats);
            perf = perform(net, nnTar, Y);
            fprintf('Number of hidden nodes per layer =  %d , 0 , 0 and the error is = %e \n', i, perf);
            if(perf<MinError)
                MinError = perf;
                MinI = i;
                MinJ = 0;
                Minz = 0;
            end
end

% 2 Hidden Layers
for i = 1: 10
    for j = 1:10
            net = newff(NormFeats, nnTar, [i j]);
            net.divideParam.trainRatio = 0.75;
            net.divideParam.testRatio = 0.25;
            net.divideParam.valRatio = 0.0;
            net.trainParam.epochs = 100;

            net =train(net, NormFeats, nnTar);
            Y = net(NormFeats);
            perf = perform(net, nnTar, Y);
            fprintf('Number of hidden nodes per layer =  %d , %d , 0 and the error is = %e \n', i,j, perf);
            if(perf<MinError)
                MinError = perf;
                MinI = i;
                MinJ = j;
                Minz = 0;
            end
    end
end

%3 Hidden layers
for i = 1: 10
    for j = 1:10
        for z = 1:10
            net = newff(NormFeats, nnTar, [i j z]);
            net.divideParam.trainRatio = 0.75;
            net.divideParam.testRatio = 0.25;
            net.divideParam.valRatio = 0.0;
            net.trainParam.epochs = 100;

            net =train(net, NormFeats, nnTar);
            Y = net(NormFeats);
            perf = perform(net, nnTar, Y);
            fprintf('Number of hidden nodes per layer =  %d , %d , %d and the error is = %e \n', i,j,z, perf);
            if(perf<MinError)
                MinError = perf;
                MinI = i;
                MinJ = j;
                Minz = z;
            end
        end
    end
end

fprintf(' the best performance was achieved when: \n Number of hidden nodes per layer =  %d , %d , %d and the error = %e \n', MinI,MinJ,Minz, MinError);
