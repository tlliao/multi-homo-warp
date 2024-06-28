function [matches_1, matches_2] = fundamentalRANSAC(pts1, pts2, parameters)   
% using fundamental matrix for robust fitting

minPtNum=8;  % minimal number of points to estimate H and e
iterNum=2000;  % maximum iterations
thDist=parameters.thDist;  % distance threshold

% delete duplicate feature match
[~,  ind1] = unique(pts1', 'rows');
[~,  ind2] = unique(pts2', 'rows');
ind = intersect(ind1, ind2);
pts1 = pts1(:, ind);
pts2 = pts2(:, ind);

%% perform coordinate normalization
ptNum = size(pts1, 2);  % number of points
[normalized_pts1, ~] = normalise2dpts([pts1; ones(1,ptNum)]);
normalized_pts1 = normalized_pts1(1:2,:);
[normalized_pts2, ~] = normalise2dpts([pts2; ones(1,ptNum)]);
normalized_pts2 = normalized_pts2(1:2,:);
points = [normalized_pts1', normalized_pts2'];

fitmodelFcn = @(points)calcFund(points); % fit function 
evalmodelFcn = @(F, points)calcDistofF(F, points);

rng(0);
[~, pro_inlierIdx] = ransac(points,fitmodelFcn,evalmodelFcn,minPtNum,thDist,'MaxNumTrials',iterNum);

inliers1 = pts1(:, pro_inlierIdx);
inliers2 = pts2(:, pro_inlierIdx);

matches_1 = inliers1;
matches_2 = inliers2;



end

function [ F ] = calcFund(points) % estimate H_inf and e' via DLT

npts1 = points(:, 1:2);
npts2 = points(:, 3:4);

%% calculation the fundamental matrix
xi = npts1(:,1); 
yi = npts1(:,2);
xi_= npts2(:,1); 
yi_= npts2(:,2);
Equation_matrix = [xi.*xi_, yi.*xi_, xi_, xi.*yi_, yi.*yi_, yi_, xi, yi, ones(size(points,1),1)];

[~,~,v] = svd(Equation_matrix, 0);
norm_F=reshape(v(1:9, 9), 3, 3)';
F=norm_F(:);

end

function dist = calcDistofF(F, points) % calculate the projective error

pts1 = points(:, 1:2)';
pts2 = points(:, 3:4);

norm_F = reshape(F,3,3);

error=[pts2, ones(size(points,1),1)]*norm_F*[pts1; ones(1,size(points,1))];

dist=abs(diag(error));

end

