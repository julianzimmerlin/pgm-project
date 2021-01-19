function out = getCliques(A,blocksize,stepsize)

%// Store blocksizes
nrows = blocksize(1);
ncols = blocksize(2);

%// Store stepsizes along rows and cols
d_row = stepsize(1);
d_col = stepsize(2);

%// Get sizes for later usages
[m,n,r] = size(A);

%// Start indices for each block
start_ind = reshape(bsxfun(@plus,[1:d_row:m-nrows+1]',[0:d_col:n-ncols]*m),[],1); %//'

%// Row indices
lin_row = permute(bsxfun(@plus,start_ind,[0:nrows-1])',[1 3 2]);  %//'

%// 2D linear indices
lidx_2D = reshape(bsxfun(@plus,lin_row,[0:ncols-1]*m),nrows*ncols,[]);

%// 3D linear indices
lidx_3D = bsxfun(@plus,permute(lidx_2D,[1 3 2]),m*n*(0:r-1));

%// Final 2D linear indices
lidx_2D_final = reshape(lidx_3D,[],size(lidx_2D,2));

%// Get linear indices based on row and col indices and get desired output
out = A(lidx_2D_final);

return;