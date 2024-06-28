function [t_error]=calctransferError(H, pts1, pts2)
%	H projects pts1 to pts2, then calcultate the symmetric transfer error between
%	pts1 and pts2

n=size(pts1,2);
pts3=H*[pts1;ones(1,n)];
pts3=pts3(1:2,:)./repmat(pts3(3,:),2,1);
d1=sqrt(sum((pts2-pts3).^2,1));

pts4=H\[pts2;ones(1,n)];
pts4=pts4(1:2,:)./repmat(pts4(3,:),2,1);
d2=sqrt(sum((pts1-pts4).^2,1));

t_error=(d1+d2)./2;

end