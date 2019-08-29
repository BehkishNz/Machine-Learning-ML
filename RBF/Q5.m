clc, close all

% again x as the problem describes
x = [-6 -5 -2 0 1 3 5];
FX=sqrt(abs(x)) .* sin((pi /2) * x); 
plot(x, FX, '+-')

% I used the variable name spread which corresponds to sigma

spread = 6; % Change that to 1, 0.5 , 3 , 6
% create a neural network
net = newrbe(x,FX,spread);

% view net
nnet.guis.closeAllViews, view(net)

xinterp = -6:0.01:6; 
Y = net(xinterp);
point = net(3.6); % for point 3.6 
fprintf('for spread =  %f, the value for point 3.6 = %f \n', spread, point);


% plot network response
hold on, plot(xinterp,Y,'r--')
legend('original function','interpolated function')
