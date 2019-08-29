clear; clc; close all

x1 = linspace(-1, 1, 500);
x2 = linspace(-4, 4, 500);

FX = sin(2*pi.*x1) .* cos(0.5*pi.*x2).* exp( -x1.^2);

net =  feedforwardnet(20);
% net.trainFcn = 'traingdx';

net.divideParam.trainRatio = 0.7;
net.divideParam.testRatio = 0.3;
net.divideParam.valRatio = 0.0;

net = train(net, [x1; x2], FX);

Y = net([x1; x2]);

perf = perform(net, FX, Y)

view(net)
%%
close all

figure,  hold on, plot(FX, 'b'), plot(Y, 'k--')
legend('Target', 'Output')