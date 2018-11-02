noHiddenNeurons = 500;
noEpochs = 30;
learningRate = 0.25;
learningDecay = 0.8;

rand();
rstate = rand("state");
save rnd_state.txt rstate
%load rndstate.txt 
%rand("state", rndstate);

% no normalization
[hlnn olnn] = crann(rows(tvec), noHiddenNeurons, 10);
trainError = zeros(1, noEpochs);
testError = zeros(1, noEpochs);
trReport = [];

printf("Learning perceptron on MNIST... \nPCA Components: %i\nHidden neurons: %i\nLearning rate:  %i\nLearning decay: %i\nPress Ctrl+C or close progress bar to stop learning.\n\n", 
       rows(tvec) - 1,
       noHiddenNeurons,
       learningRate,
       learningDecay); fflush(stdout);
       
for epoch=1:noEpochs
	tic();
	[hlnn olnn terr] = backprop(tvec, tlab, hlnn, olnn, learningRate);
  clsRes = anncls(tstv, hlnn, olnn);
  cfmx = confMx(tstl, clsRes);
  errcf2 = compErrors(cfmx);
	epochTime = toc();
  accuracy = round(10000 * (1 - errcf2(2))) / 100;
  learningRate = learningRate * learningDecay;
  
  printf("Epoch %i, accuracy = %i%%, learning time %i s. \n", epoch, accuracy, epochTime);
	fflush(stdout);
end

clsRes = anncls(tstv, hlnn, olnn);
cfmx = confMx(tstl, clsRes)
errcf2 = compErrors(cfmx)

