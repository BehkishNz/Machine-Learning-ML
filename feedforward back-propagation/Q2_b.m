clear; clc
x1 = linspace(-1, 1, 100);
FX1 = x1 .* sin(6 * pi * x1) .* exp(-x1.^2);

x2 = linspace(-2, 2, 100);
FX2 = exp(-x2 .^ 2) .* atan(x2) .* sin(4 .* pi* x2); 

MinError = 100;
MinI = 0;
for i = 1:100
    net =  feedforwardnet(i);
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.0;
    net.divideParam.testRatio = 0.3;

    %net = train(net, x1, FX1);
    net = train(net, x2, FX2);
    %Y = net(x1);
    %perf = perform(net, FX1, Y);
    Y = net(x2);
    perf = perform(net, FX2, Y);
        fprintf('Number of hiden nodes =  %d and the error is = %f \n', i, perf);
    if(perf<MinError)
        MinError = perf;
        minI = i;
    end
end

 net =  feedforwardnet(minI);
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.0;
    net.divideParam.testRatio = 0.3;

    %net = train(net, x1, FX1);
    net = train(net, x2, FX2);
    %Y = net(x1);
    %perf = perform(net, FX1, Y);
    Y = net(x2);
    perf = perform(net, FX2, Y)
    minI
    MinError

%%
close all
%figure,  hold on, plot(FX1, 'b'), plot(Y, 'k--')
figure,  hold on, plot(FX2, 'b'), plot(Y, 'k--')
legend('Target', 'Output')