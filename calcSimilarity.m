function [Similar] = calcSimilarity(matches1, matches2)
% calculate the similarity transformation between matches1 and matches2
num_pts=size(matches1,2);
[data_normpts1, T1] = normalise2dpts([matches1; ones(1,num_pts)]);
[data_normpts2, T2] = normalise2dpts([matches2; ones(1,num_pts)]);
A1=reshape(data_normpts1(1:2,:),[],1);
A2=reshape([-data_normpts1(2,:);data_normpts1(1,:)],[],1);
A34=repmat([1 0; 0 1],num_pts,1);
b=reshape(data_normpts2(1:2,:),[],1);
beta=[A1,A2,A34]\b;
S=[beta(1) -beta(2) beta(3);
    beta(2)  beta(1) beta(4);
    0        0       1];
Similar=T2\(S*T1);

end