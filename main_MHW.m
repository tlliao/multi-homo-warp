clear; clc; close all; 
%% Setup VLFeat toolbox.
%----------------------
addpath('modelspecific'); 
addpath('multi-model fitting\')
addpath('multi-model fitting\bin\');
addpath('multi-model fitting\cplus_files\');
addpath('multi-model fitting\GCO_files\');
run vlfeat-0.9.21/toolbox/vl_setup;

% setup parameters
% Parameters of SIFT detection
parameters.peakthresh=0;
parameters.edgethresh=500;

parameters.thDist=0.05;  % distance threshold

%%---figure display or not
parameters.display=false;
parameters.dist = 5; % distance threshold for feature matches refining of each model

%-- % Parameters in multi-homography fitting
parameters.lambda=20; % The coefficient of smoothcost in energy function
parameters.beta=10; % The coefficient of label cost in energy function
parameters.maxdatacost=1e4;
parameters.gamma=2e2; % distance between feature point and outlier model (datacost)

imgpath='Imgs/';

path1=sprintf('%s%s',imgpath,'3_l.jpg'); %
path2=sprintf('%s%s',imgpath,'3_r.jpg'); %
img1=im2double(imread(path1));  % target image
img2=im2double(imread(path2));  % reference image

%% segment angthing model results input
labels1=double(imread([imgpath, 'segmentation_3_l.png']));

%% feature detection and matching, multi-homography initialization
[pts1, pts2]=siftMatch(img1, img2, parameters);

[matches_1, matches_2]=fundamentalRANSAC(pts1, pts2, parameters);
[init_H, ~]=multiHomoGeneraton(matches_1, matches_2);


%%  multi-homography fitting via PEaRL method
[multi_homos, cell_matches]=multiHomoFitting(matches_1,matches_2,img1,img2,labels1,init_H,parameters);
fprintf('number of homographies: %d\n', size(multi_homos,1));

[final_labels1, final_homos, overlapped_C]=sammaskLabeling(img1, img2, labels1, multi_homos, cell_matches);

[panorama, warped_img1, warped_img2, ssim_index, psnr_index]=samBackwardMapping(img1, img2, final_homos, final_labels1, overlapped_C);
