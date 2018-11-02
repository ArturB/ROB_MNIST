function lab = anncls(sample, hidlw, outlw)
% simple ANN classifier
% sample - data to be classified (a column vector)
% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix

% lab - classification result (index of output layer neuron with highest value)
% ATTENTION: we assume that constant value IS NOT INCLUDED in tset rows

	[~, lab] = max( annout(sample, hidlw, outlw), [], 1);
