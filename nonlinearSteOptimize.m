function [ste_homo] = nonlinearSteOptimize(pts1,pts2)
% estimate the homography by optimizing symmetric transfer error between
% pts1 and pts2
%% calculate optimal H
%options = optimoptions('lsqnonlin','Display','iter');
% options.Algorithm = 'levenberg-marquardt';
% options.Display = 'iter';
% options.FunctionTolerance = 1e-8;
% options.MaxIterations = 1e3;
h0=calcHomo(pts1,pts2);
h0=h0(:);
options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt', 'Display', 'off', 'FunctionTolerance', 1e-8, 'MaxIterations', 1e3);
[x, ~] = lsqnonlin(@(x)calcSte(x, pts1, pts2), h0,[],[],options);
ste_homo=reshape(x,3,3);


end


function [t_error]=calcSte(x, pts1, pts2)
%	H projects pts1 to pts2, then calcultate the symmetric transfer error between
%	pts1 and pts2
H=reshape(x,3,3);
n=size(pts1,2);
pts3=H*[pts1;ones(1,n)];
pts3=pts3(1:2,:)./repmat(pts3(3,:),2,1);
d1=sqrt(sum((pts2-pts3).^2,1));

pts4=H\[pts2;ones(1,n)];
pts4=pts4(1:2,:)./repmat(pts4(3,:),2,1);
d2=sqrt(sum((pts1-pts4).^2,1));

t_error=sqrt((d1+d2)./2);

end