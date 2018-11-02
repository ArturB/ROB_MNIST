fnames = { 'train-images.idx3-ubyte'; 'train-labels.idx1-ubyte';  
				't10k-images.idx3-ubyte'; 't10k-labels.idx1-ubyte' };

[tlab tvec] = readmnist(fnames{1,1}, fnames{2,1});
[tstl tstv] = readmnist(fnames{3,1}, fnames{4,1});

tlab += 1;
tstl += 1;

warning ("off", "Octave:broadcast");

% remove columns with zero std 
toRemain = std(tvec) != 0;
tvec = tvec(:, toRemain);
tstv = tstv(:, toRemain);

% calculate PCA transform
[mupca trmxpca] = prepTransform(tvec, 400);
tvec = pcaTransform(tvec, mupca, trmxpca);
tstv = pcaTransform(tstv, mupca, trmxpca);

% proper vector must be a column, by definition. Also, input vector must contain a bias
tvec = [tvec ones(rows(tvec),1)];
tvec = tvec';
tstv = [tstv ones(rows(tstv),1)];
tstv = tstv';

% store data for normalization
mu = mean(tvec);
sig = std(tvec);