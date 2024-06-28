function [init_H, cell_matches]=multiHomoGeneraton(pts1, pts2)
% Use RANSAC-based method to find the initial models(homographies) for
% Multi-homography fitting
%% PROPOSE Initial finite Models (Homographies) Cell_H_0 based on RANSAC
iter_num=1000;
init_H=cell(1,round(length(pts1)/4)); % initial homographies
cell_matches=cell(round(length(pts1)/4),2);
[matches1, matches2]=homoRANSAC(pts1,pts2,iter_num);
init_H{1}=calcHomo(matches1, matches2);
cell_matches{1,1}=matches1;
cell_matches{1,2}=matches2;
k=2;
[rest_pts1, ia]=setdiff(pts1', matches1', 'rows');
rest_pts1=rest_pts1';
rest_pts2=pts2(:,ia);
while length(rest_pts1)>=50
    iter_num=max(round(iter_num/2),2*length(rest_pts1));
    [matches1, matches2]=homoRANSAC(rest_pts1,rest_pts2,iter_num);
    init_H{k}=calcHomo(matches1, matches2);
    cell_matches{k,1}=matches1;
    cell_matches{k,2}=matches2;   
    [rest_pts1, ia]=setdiff(rest_pts1', matches1', 'rows');
    rest_pts1=rest_pts1';
    rest_pts2=rest_pts2(:,ia);
    k=k+1; 
end
init_H=init_H(1:k-1);
cell_matches=cell_matches(1:k-1,:);

end

