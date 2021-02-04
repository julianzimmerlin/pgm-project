function alphas = learnAlphas(filters,f_imgs)
clc;
rng(0);
num_samples = 500;
% calculate alphas for all filters
% filters = filters(:,1:end-1);
% initialize alphas to zero
alphas = zeros(size(filters,2),1);
patchsize = 7;
path = '.\BSR_bsds500\BSR\BSDS500\data\images\train\';
path2 = '.\BSR_bsds500\BSR\BSDS500\data\images\test\';
for i=1:numel(f_imgs) % loop through training images and extract num_samples patches from each of them
    disp( ['Current file is: ' f_imgs(i).name] );
    if i <= 200
        img = imread([path f_imgs(i).name]);
    else
        img = imread([path2 f_imgs(i).name]);
    end
    patches = zeros(patchsize,patchsize,3,num_samples);
    for p=1:num_samples
        x = randi(size(img,1)-patchsize);
        y = randi(size(img,2)-patchsize);
        patches(:,:,:,p) = img(x:x+6, y:y+6, :);
    end
    all_patches(:,:,:,(i-1)*num_samples+1:i*num_samples) = patches;
end
learning_rate = 1; % 0.001; we can still adjust this afterwards by rescaling alphas
% Perform one step of gradient ascent to learn alphas
alphas = alphas + learning_rate * getGradient(filters,all_patches);
