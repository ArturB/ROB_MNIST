function olout = annout(sample, hidlw, outlw)

	hlact = hidlw * sample;
	hlout = actf(hlact);
	
	%olact = outlw * hlout;
  olact = outlw * hlout;
	olout = actf(olact);

end
