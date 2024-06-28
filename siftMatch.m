function [pts1, pts2] = siftMatch( img1, img2, parameters )
%--------------------------------------
% SIFT keypoint detection and matching.
%--------------------------------------
peakthresh = parameters.peakthresh;
edgethresh = parameters.edgethresh;
%fprintf('  Keypoint detection and matching...');tic;
[ kp1,ds1 ] = vl_sift(single(rgb2gray(img1)),'PeakThresh', peakthresh,'edgethresh', edgethresh);
[ kp2,ds2 ] = vl_sift(single(rgb2gray(img2)),'PeakThresh', peakthresh,'edgethresh', edgethresh);
matches   = vl_ubcmatch(ds1, ds2);
%fprintf('done (%fs)\n',toc);

% extract match points' position
pts1 = kp1(1:2,matches(1,:));  pts2 = kp2(1:2,matches(2,:)); 

% delete duplicate feature match
[~,  ind1] = unique(pts1', 'rows');
[~,  ind2] = unique(pts2', 'rows');
ind = intersect(ind1, ind2);
pts1 = pts1(:, ind);
pts2 = pts2(:, ind);

end

