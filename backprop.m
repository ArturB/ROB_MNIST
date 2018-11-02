function [hidlw outlw terr] = backprop(tset, tslb, inihidlw, inioutlw, lr)
% derivative of sigmoid activation function
% tset - training set (every column represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate

% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN

	% 1. Set output matrices to initial values
	hidlw = inihidlw;
	outlw = inioutlw;
	
	% 2. Set total error to 0
	terr = 0;
  
  % samples number
  smpnum = columns(tset);
	
  progress_bar = waitbar(0, 'Learning perceptron...');
	% foreach sample in the training set
	for i=1:smpnum
		% vector of network input (hidden layer input)
		inp = tset(:,i);
		% vector of hidden layer output (output layer input)
		inter = actf( hidlw * inp );
		% vector of network output
		out = annout(inp,hidlw,outlw);
		% vector of expected network output
		expOut = zeros(rows(outlw),1);
		expOut(tslb(i)) = 1;
		% vector of network error
		outerr = expOut .- out;
		% value of total squarred error of ANN
		terr = sum(outerr .* outerr);
		% output weights change
		incOutW = lr * (outerr .* inter') .* actdf( out );
		% hidden weights error
		hiderr = outlw' * (outerr .* out);
		% hidden weights change
		incHidW = lr * (hiderr .* inp') .* actdf( inter ); 
		% update weights
		hidlw += incHidW;
		outlw += incOutW;

		% update waitbar
		if(mod(i, 500) == 0)
      waitbar(i/smpnum, progress_bar);
    end

	end

  close(progress_bar);
  
end