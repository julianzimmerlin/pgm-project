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
sigma = 15;
N = I + sigma * randn(size(I));

% We rescale the alphas here instead of adjusting the learning rate
% so that we dont't have to retrain them every time
% 5e-10 seems best

%factors = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8];
%factors = [2.5e-10, 5e-10, 1e-9, 2e-9, 4e-9];
factors = [4e-10, 5e-10, 6e-10];

results = zeros(size(factors, 2));
for i = 1:size(factors,2)
    alphas_scaled = alphas * factors(i);
    % Perform 100 iterations of denoising using McAuley's inference implementation
    O = denoise_foe(N, filters, mirrorfilters, alphas_scaled, sigma, 100, 62.5, I);
    imshow(uint8(O));
    results(i) = psnr(O,I,255);
end
for i = 1:size(factors,2)
    disp(['Final PSNR for factor ' num2str(factors(i)) ': ' num2str(results(i))])
end