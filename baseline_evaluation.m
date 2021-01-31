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
load('combinedFilters3.mat')
load('alphas_combined3.mat');
% Remove last filter (gray image)
alphas = alphas_combined3(1:end-1);
filters = combinedFilters3(:,1:end-1);
%Reshape our filters to match their format
mirrorfilters = filters(size(filters,1):-1:1, :);
filters = reshape(filters, cliquesize, cliquesize, 3, []); 
mirrorfilters = reshape(mirrorfilters, cliquesize, cliquesize, 3, []);

ganResults3 = zeros(2,2,numel(f_imgs)); % num_methods * num_noise_models * num_images
% Iterate over test images
for i=1:numel(f_imgs)
    disp(['Round ' num2str(i) ' --------------------------------'])
    disp( ['Current file is: ' f_imgs(i).name] );
    I = double(imread([path f_imgs(i).name]));
    
    % Add Gaussian noise equally to all channels
    sigma = 25;
    N_1 = I + sigma * randn(size(I));
    N_1(N_1>255) = 255;
    N_1(N_1<0) = 0;
    
    % Add unequal noise to each channel
    sigma_r = 128;
    sigma_g = 15;
    sigma_b = 5;
    factors = zeros(size(I));
    factors(:,:,1) = sigma_r * ones(size(I,1,2));
    factors(:,:,2) = sigma_g * ones(size(I,1,2));
    factors(:,:,3) = sigma_b * ones(size(I,1,2));
    N_2 = I + factors .* randn(size(I));
    N_2(N_2>255) = 255;
    N_2(N_2<0) = 0;
    
    % display PSNR of noisy image
    disp('Noisy Images')
    disp(['Equal noise: ' num2str(psnr(N_1,I)) ', unequal noise: ' num2str(psnr(N_2,I))])
    
    % denoise using channel independent baseline
   
    disp('Independent baseline')
    eta=200; % check if this is good
    [Out_1, ~] = denoising_grad_ascent_student_color(N_1, 100, eta, 100, N_1,1);
    [Out_2, ~] = denoising_grad_ascent_student_color(N_2, 100, eta, 100, N_2,1);
    %figure, plot(1:size(log_post,2),log_post(1,:));
    %figure, plot(1:size(log_post,2),log_post(2,:));
    %figure, plot(1:size(log_post,2),log_post(3,:));
    ganResults3(1,1,i) = psnr(Out_1,I);
    ganResults3(1,2,i) = psnr(Out_2,I);
    disp(['Equal noise: ' num2str(ganResults3(1,1,i)) ', unequal noise: ' num2str(ganResults3(1,2,i))])
    
    % denoise using learned filters
    disp('Learned filters')
    % Perform 100 iterations of denoising using McAuley's inference implementation
    O_1 = denoise_foe(N_1, filters, mirrorfilters, alphas, sigma, 1e9, 150, 1e-8, I); % alphas .* 4e-10 % delta_t formerly 6.5
    O_2 = denoise_foe(N_2, filters, mirrorfilters, alphas, [sigma_r sigma_g sigma_b], 5e8, 200, 4e-8, I); % alphas .* 1e-9
    ganResults3(2,1,i) = psnr(O_1,I);
    ganResults3(2,2,i) = psnr(O_2,I);
    disp(['Equal noise: ' num2str(ganResults3(2,1,i)) ', unequal noise: ' num2str(ganResults3(2,2,i))])
    
    figure, imshow(uint8(N_2));
    figure, imshow(uint8(O_2));
end

save('ganResults3.mat', 'ganResults3')