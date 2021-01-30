clc;
clear all;
path = '.\BSR_bsds500\BSR\BSDS500\data\images\train\';
path2 = '.\BSR_bsds500\BSR\BSDS500\data\images\test\';
rng(0)
f = dir(path);
f2 = dir(path2);
f_imgs = struct([]);
patchsize = 3;
num_samples = 500; % per training image
all_patches = zeros(num_samples*(200), 3*patchsize^2,1);
j=1;
for i=1:numel(f) % ignore files that aren't jpg images
    [~,~, fExt] = fileparts(f(i).name);
    if strcmpi(fExt,'.jpg')
        f_imgs(j).name = f(i).name;
        j = j+1;
    end
end
%for i=1:numel(f2) % ignore files that aren't jpg images
%    [~,~, fExt] = fileparts(f2(i).name);
%    if strcmpi(fExt,'.jpg')
%        f_imgs(j).name = f2(i).name;
%        j = j+1;
%    end
%end
for i=1:numel(f_imgs) % loop through training images and extract num_samples patches from each of them
    disp( ['Current file is: ' f_imgs(i).name] );
    if i <= 200
        img = imread([path f_imgs(i).name]);
    else
        img = imread([path2 f_imgs(i).name]);
    end
    img = rgb2ycbcr(img);
    patches = zeros(num_samples,3*patchsize^2);
    for p=1:num_samples
        x = randi(size(img,1)-patchsize);
        y = randi(size(img,2)-patchsize);
        patches(p,:) = reshape(img(x:x+2, y:y+2, :), 3*patchsize^2,1);
    end
    all_patches((i-1)*num_samples+1:i*num_samples, :) = patches;
end

cov_matrix = cov(all_patches);
[V,D] = eig(cov_matrix);
save('filters_ycbcr.mat', 'V');
%%
alphas = learnAlphas(V,f_imgs);
save('alphas_ycbcr.mat', 'alphas');
disp('end')
%% Learning alphas when using cnn filters
% load cnn filter
%V_cnn =  double(load('cnn3Filters.mat').mydata);
%alphas_cnn = learnAlphas(V_cnn,f_imgs);
%save('alphas_cnn3.mat', 'alphas_cnn');
%disp('end')