function err = test_ann(tvec, tlab, tstv, tstl, c_hid, comp_count = 40)

	printf("Calculating %i first PCA components...\n", comp_count); fflush(stdout);
  
	% calculating PCA-transformed data...
	[mu trmx] = prepTransform(tvec, comp_count);
	tvec = pcaTransform(tvec, mu, trmx);
	tstv = pcaTransform(tstv, mu, trmx);
  
	printf("PCA components calculated!\nTraining perceptron...\n"); fflush(stdout);

	c_feat = columns(tvec);
	c_class = rows(unique(tlab));

	[hw ow] = crann(c_feat, c_hid, c_class);

  epchsNum = 10;
	whand = waitbar(0, "Training perceptrons 0%");
	prog = 0;
	fullProg = epchsNum * rows(tvec);

	terr = 0;
	lr = 1;
	for i = 1:epchsNum
		
		% foreach sample in the training set
		for i=1:rows(tvec)
			% vector of network input (hidden layer input)
			inp = tvec(i,:)';
			% vector of hidden layer output (output layer input)
			inter = actf( hw * [inp; 1] );
			% vector of network output
			out = annout(inp,hw,ow);
			% vector of expected network output
			expOut = zeros(rows(ow),1);
			expOut(tlab(i)) = 1;
			% vector of network error
			outerr = expOut .- out;
			% value of total squarred error of ANN
			terr = sum(outerr .* outerr);
			% out weights change
			incOutW = lr * ( outerr .* inter' );
			% hidden weights error
			hiderr = ow' * outerr;
			% hidden weights change
			incHidW = lr * ( hiderr .* [inp; 1]'); 
			% update weights
			hw += incHidW;
			ow += incOutW;

			% update waitbar
			prog++;
      if mod(prog, fullProg / 100) == 0
        title = ["Training perceptron " mat2str(round(100 * prog / fullProg)) " %"];
        waitbar(prog / fullProg, whand, title);
      end
		end
	end

	hw;
	ow;

	close(whand);

	printf("Perceptron trained!\nCalculating confusion matrix...\n"); fflush(stdout);

	clsRes = zeros(size(tstl));
	for i = 1:rows(clsRes)
		clsRes(i) = anncls(tstv(i,:)', hw, ow);
	end

	cfmx = confMx(tstl, clsRes)
	errcf2 = compErrors(cfmx)

	printf("All done!\n"); 

end
