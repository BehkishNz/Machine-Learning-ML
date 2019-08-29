function [classes] = mat2label(a)
%%
numClass = max(a(:));
ar = repmat(a, numClass,1);
b = ones(size(ar));

for i = 1:numClass
    b(i,:) = i * b(i,:);
end
classes = +(ar == b);
end