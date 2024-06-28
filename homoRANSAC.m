function [matches_1, matches_2] = homoRANSAC(pts1, pts2, iterNum)   
% using fundamental matrix for robust fitting

minPtNum=4;  % minimal number of points to estimate H and e
thDist=0.01;  % distance threshold

%% perform coordinate normalization
[normalized_pts1, ~] = normalise2dpts([pts1; ones(1,size(pts1, 2))]);
[normalized_pts2, ~] = normalise2dpts([pts2; ones(1,size(pts2, 2))]);
points = [normalized_pts1', normalized_pts2'];

fitmodelFcn = @(points)calcNormHomo(points); % fit function 
evalmodelFcn = @(homo, points)calcDistofHomo(homo, points);

rng(0);
[~, inlierIdx] = ransac(points,fitmodelFcn,evalmodelFcn,minPtNum,thDist,'MaxNumTrials',iterNum);

inliers1 = pts1(:, inlierIdx);
inliers2 = pts2(:, inlierIdx);

matches_1 = inliers1;
matches_2 = inliers2;

end

function [ homo ] = calcNormHomo(points) % estimate H_inf and e' via DLT

npts1 = points(:, 1:3)';
npts2 = points(:, 4:6)';

%% calculation the initial H0 and e0
Equation_matrix = zeros(2*size(npts1, 2), 9);
for i=1:size(npts1, 2)
    xi = npts1(1,i); yi = npts1(2,i);
    xi_= npts2(1,i); yi_= npts2(2,i);
    tmp_coeff1 = [xi, yi, 1, 0,  0,  0, -xi*xi_, -yi*xi_, -xi_];
    tmp_coeff2 = [0,  0,  0, xi, yi, 1, -xi*yi_, -yi*yi_, -yi_];
    Equation_matrix(2*i-1:2*i, :) = [tmp_coeff1; tmp_coeff2];
end

[~,~,v] = svd(Equation_matrix, 0);
norm_homo = reshape(v(1:9, end), 3, 3)';
homo = norm_homo(:);

end

function dist = calcDistofHomo(homo, points) % calculate the projective error

pts1 = points(:, 1:3)';
pts2 = points(:, 4:6)';

H = reshape(homo(1:9),3,3);

tmp1 = (H(1,:)*pts1)./pts1(3,:);
tmp2 = (H(2,:)*pts1)./pts1(3,:);
tmp3 = (H(3,:)*pts1)./pts1(3,:);
mapped_pts2(1,:) = tmp1./tmp3;
mapped_pts2(2,:) = tmp2./tmp3;
dist = sqrt(sum((mapped_pts2-pts2(1:2,:)).^2, 1));

end

