function [ valid_flag ]=homoValidJudge(homo, size_img)

% judge whether the homography implies a valid warped target image
% the warped quadrangle should be convex after warping

ori_quad = [1, size_img(2), size_img(2), 1; 1, 1, size_img(1), size_img(1)]';
w_pts=homo*[ori_quad'; ones(1,4)];
w_pts=w_pts(1:2,:)./repmat(w_pts(3,:),2,1);

warp_quad=[w_pts, w_pts(:,1)]';
valid_flag=true;
for i=1:4
    line_pts=[warp_quad(i,:); warp_quad(i+1,:)];
    diff_pts=setdiff(warp_quad, line_pts, 'rows');
    valid_flag_i=pointLinePosition(line_pts,diff_pts);
    valid_flag = valid_flag && valid_flag_i;
end

% figure,plot([ori_quad(:,1); ori_quad(1,1)],[ori_quad(:,2);ori_quad(1,2)]);
% hold on
% plot(warp_quad(:,1),warp_quad(:,2));
% hold off


end


function valid_flag=pointLinePosition(line_pts, point_pts)
% given a line and two points, if the two points lie in the same side of
% the line, valid_flag is true, otherwise, it is false
% plot(line_pts(:,1),line_pts(:,2));
% hold on
% plot(point_pts(:,1),point_pts(:,2),'rx');
% hold off

a=line_pts(2,2)-line_pts(1,2);  % y2-y1
b=line_pts(1,1)-line_pts(2,1);  % x1-x2
c=line_pts(2,1)*line_pts(1,2)-line_pts(1,1)*line_pts(2,2);  % x2*y1-x1*y2
a1=a/sqrt(a^2+b^2+c^2);
b1=b/sqrt(a^2+b^2+c^2);
c1=c/sqrt(a^2+b^2+c^2);
flag1=a1*point_pts(1,1)+b1*point_pts(1,2)+c1;
flag2=a1*point_pts(2,1)+b1*point_pts(2,2)+c1;
valid_flag=flag1*flag2>0;

end