clc;
clear all;
path = '.\BSR_bsds500\BSR\BSDS500\data\images\train\';
rng(0)
f = dir(path);
f_imgs = struct([]);
patchsize = 3;
num_samples = 500; % per training image
all_patches = zeros(num_samples*200, 3*patchsize^2,1);
j=1;
for i=1:numel(f) % ignore files that aren't jpg images
    [~,~, fExt] = fileparts(f(i).name);
    if strcmpi(fExt,'.jpg')
        f_imgs(j).name = f(i).name;
        j = j+1;
    end
end
for i=1:numel(f_imgs) % loop through training images and extract num_samples patches from each of them
    disp( ['Current file is: ' f_imgs(i).name] );
    img = imread([path f_imgs(i).name]);
    
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
save('filters.mat', 'V');
%%
alphas = learnAlphas(V,f_imgs);
disp('end')