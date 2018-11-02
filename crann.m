function [hl ol] = crann(cfeat, chn, cclass)
% generates hidden and output ANN weight matrices
% cfeat - number of features 
% chn - number of neurons in the hidden layer
% cclass - number of neurons in the outpur layer (= number of classes)

% hl - hidden layer weight matrix
% ol - output layer weight matrix

% ATTENTION: we assume that constant value (bias) IS NOT INCLUDED

  hl = (rand(chn, cfeat) .- 0.5)  ./ sqrt(cfeat);
	ol = (rand(cclass, chn) .- 0.5) ./ sqrt(chn);

end