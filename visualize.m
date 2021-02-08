% Visualize SVD Filters
clc;clear all;
load('filters_200_500.mat');
filters = reshape(V,3,3,3,[]);
% normalize each filter in the range 0 to 1
for i = 1 : size(filters,4)
    img = filters(:,:,:,i);
    img = (img - min(img(:))) ./ (max(img(:))-min(img(:)));
    filters(:,:,:,i) = img;
end
figure;
montage(filters,'Size',[3 9],'BorderSize',[2 2],'BackgroundColor','white');
%%
load('gan_filters_1.mat');
ganFilters = gan_filters;
load('gan_filters_2.mat');
ganFilters = [ganFilters,gan_filters];
ganFilters = reshape(ganFilters,3,3,3,[]);
% normalize each filter in the range 0 to 1
for i = 1 : size(ganFilters,4)
    img = ganFilters(:,:,:,i);
    img = (img - min(img(:))) ./ (max(img(:))-min(img(:)));
    ganFilters(:,:,:,i) = img;
end

figure;
montage(ganFilters,'Size',[3 9],'BorderSize',[2 2],'BackgroundColor','white');
%%
load('alphas_200_500.mat');
figure;
plot((1:26),flip(alphas(1:26)),'r-','LineWidth',2);
xlim([1 26])
ylim([min(alphas(:)) max(alphas(:))])
xlabel('Filter Rank');
ylabel('Alphas');

