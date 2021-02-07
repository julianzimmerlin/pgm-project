clear all;
close all;
path = '.\images\';
addpath('inference')
addpath('inference_baseline')
rng(0);

% load test images
f = dir(path);
f_imgs = struct([]);
j=1;
for i=1:numel(f) % ignore files that aren't jpg images
    [~,~, fExt] = fileparts(f(i).name);
    if strcmpi(fExt,'.jpg')
        f_imgs(j).name = f(i).name;
        j = j+1;
    end
end

% load filters and alphas
cliquesize=3;
%load('filters_200_500.mat');
%load('alphas_200_500.mat');
load('filters_ycbcr.mat');
load('alphas_ycbcr.mat');
% Remove last filter (gray image)
alphas = alphas(1:end-1);
filters = V(:,1:end-1);
%Reshape our filters to match their format
mirrorfilters = filters(size(filters,1):-1:1, :);
filters = reshape(filters, cliquesize, cliquesize, 3, []); 
mirrorfilters = reshape(mirrorfilters, cliquesize, cliquesize, 3, []);

results = zeros(2,numel(f_imgs));
for i=1:numel(f_imgs)
    disp(['Round ' num2str(i) ' --------------------------------'])
    disp( ['Current file is: ' f_imgs(i).name] );
    I = double(rgb2ycbcr(imread([path f_imgs(i).name])));
    I_rgb = ycbcr2rgb(uint8(I));
    
    % Add unequal noise to each channel
    sigma_r = 25;
    sigma_g = 25;
    sigma_b = 25;
    factors = zeros(size(I));
    factors(:,:,1) = sigma_r * ones(size(I,1,2));
    factors(:,:,2) = sigma_g * ones(size(I,1,2));
    factors(:,:,3) = sigma_b * ones(size(I,1,2));
    N_2 = I + factors .* randn(size(I));
    N_2(N_2>240) = 240;
    N_2(N_2<16) = 16;
    N_2_rgb = ycbcr2rgb(uint8(N_2));
    figure, imshow(uint8(N_2_rgb));

    
    disp('Noisy Images')
    disp([ num2str(psnr(N_2_rgb,I_rgb)) ])
    
    % denoise using channel independent baseline
    disp('Independent baseline')
    eta=200; % check if this is good
    [Out_2, ~] = denoising_grad_ascent_student_color(N_2, 100, eta, 100, N_2,1);
    Out_2_rgb = ycbcr2rgb(uint8(Out_2));
    disp(num2str(psnr(Out_2_rgb,I_rgb)))
    
    % ycbcr and rgb denoising
    O_2 = denoise_foe(double(N_2), filters, mirrorfilters, alphas, [sigma_r sigma_g sigma_b], 1e7, 150, 1e-8, I_rgb); % alphas .* 1e-9
    Out_2_rgb = denoise_foe(double(N_2_rgb), filters, mirrorfilters, alphas, [sigma_r sigma_g sigma_b], 1e7, 200, 1e-8, I_rgb);
    O_2_rgb = ycbcr2rgb(uint8(O_2));
    results(1,i) = psnr(O_2_rgb,I_rgb);
    results(2,i) = psnr(Out_2_rgb,I_rgb);
    disp(['ycbcr: ' num2str(results(1,i)) ', rgb: ' num2str(results(2,i))])
end

save('results_ycbcr_vs_rgb.mat', 'results')