cliquesize=3;
addpath('inference')
load('filters.mat');
load('alphas.mat');
% Remove last filter (gray image)
alphas = alphas(1:end-1);
filters = V(:,1:end-1);

%Reshape our filters to match their format
mirrorfilters = filters(size(filters,1):-1:1, :);
filters = reshape(filters, cliquesize, cliquesize, 3, []); 
mirrorfilters = reshape(mirrorfilters, cliquesize, cliquesize, 3, []);

% Load image
I = double(imread('images/castle.jpg'));

% Add Gaussian noise
sigma_r = 128;
sigma_g = 15;
sigma_b = 5;
factors = zeros(size(I));
factors(:,:,1) = sigma_r * ones(size(I,1,2));
factors(:,:,2) = sigma_g * ones(size(I,1,2));
factors(:,:,3) = sigma_b * ones(size(I,1,2));
N = I + factors .* randn(size(I));
N(N>255) = 255;
N(N<0) = 0;

% We rescale the alphas here instead of adjusting the learning rate
% so that we dont't have to retrain them every time
% 4e-10 seems best for sigma=25, 5e-10 for sigma=15

%lambdas = [4e8,5e8,6e8];
%lambdas = [5e6, 5e7, 5e8];
lambdas = [4e8];
results = zeros(size(lambdas, 2));
for i = 1:size(lambdas,2)
    % Perform 100 iterations of denoising using McAuley's inference implementation
    O = denoise_foe(N, filters, mirrorfilters, alphas, [sigma_r sigma_g sigma_b], lambdas(i), 250, 4e-8, I);
    imshow(uint8(O));
    results(i) = psnr(O,I);
end
for i = 1:size(lambdas,2)
    disp(['Final PSNR for lambdas ' num2str(lambdas(i)) ': ' num2str(results(i))])
end