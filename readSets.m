fnames = { 'train-images.idx3-ubyte'; 'train-labels.idx1-ubyte';  
				't10k-images.idx3-ubyte'; 't10k-labels.idx1-ubyte' };

csvnames = { 'train-images.csv'; 'train-labels.csv';  
				't10k-images.csv'; 't10k-labels.csv' };

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
[mupca trmxpca] = prepTransform(tvec, 40);
tvec = pcaTransform(tvec, mupca, trmxpca);
tstv = pcaTransform(tstv, mupca, trmxpca);

% save PCA-ed images to CSV files. Each vector is - as in valid algebra - a column vector, not row. 
csvwrite(csvnames{1,1},tvec.') % train-images
csvwrite(csvnames{2,1},tlab.') % train-labels
csvwrite(csvnames{3,1},tstv.') % t10k-images (test-images)
csvwrite(csvnames{4,1},tstl.') % t10k-labels (test-labels)

% proper vector must be a column, by definition. Also, input vector must contain a bias
%tvec = [tvec ones(rows(tvec),1)];
%tvec = tvec';
%tstv = [tstv ones(rows(tstv),1)];
%tstv = tstv';

% store data for normalization
%mu = mean(tvec);
%sig = std(tvec);