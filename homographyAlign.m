function [homo1, homo2]=homographyAlign(img1,img2,init_H)
%input: target image and reference image, saliency map of the two images
%output: homography-warped target and reference, with their corresponding
%saliency maps

pt = [1, 1, size(img1,2), size(img1,2);
      1, size(img1,1), 1, size(img1,1);
      1, 1, 1, 1]; 
H_pt = init_H*pt; H_pt = H_pt(1:2,:)./repmat(H_pt(3,:),2,1);

% calculate the convas
off = round([ 1 - min([1 H_pt(1,:)]) + 1 ; 1 - min([1 H_pt(2,:)]) + 1 ]);
cw = max([size(img2,2),ceil(H_pt(1,:))])-min([1,floor(H_pt(1,:))])+1;
ch = max([size(img2,1),ceil(H_pt(2,:))])-min([1,floor(H_pt(2,:))])+1;

if cw>=5*size(img2,2) || ch>=5*size(img2,1) % for invalid homography, return original images
    homo1=img1;
    homo2=img2;
    return;
end

tform = projective2d(init_H');
img1mask = imwarp(true(size(img1,1),size(img1,2)), tform, 'nearest');

img1To2 = imwarp(img1, tform);  % 单应性变形图
img1To2 = cat(3,img1mask, img1mask, img1mask).*img1To2;

cw=max(cw,round(min(H_pt(1,:)))+off(1)-2+size(img1To2,2));
ch=max(ch,round(min(H_pt(2,:)))+off(2)-2+size(img1To2,1));

homo1 = zeros(ch,cw,3); homo2 = zeros(ch,cw,3); % 变形结果图

% 变形后待拼接图和基准图
homo1(round(min(H_pt(2,:)))+off(2)-1:round(min(H_pt(2,:)))+off(2)-2+size(img1To2,1),...
    round(min(H_pt(1,:)))+off(1)-1:round(min(H_pt(1,:)))+off(1)-2+size(img1To2,2),:) = img1To2;
homo2(off(2):(off(2)+size(img2,1)-1),off(1):(off(1)+size(img2,2)-1),:) = img2;
end

