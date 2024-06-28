function [ panorama,  warped_img1, warped_img2, ssim_index, psnr_index ] = samBackwardMapping( img1, img2, sam_H, labels, overlaped_C)
% Inputs
% [1] target image, reference image
% [2] multiple homographies for each segmented parts
% [3] segmented parts
%  Output
% [1] the final panorama via linear blending
%% initial forward mapping, calculate mapping bounding box 
sam_idx=label2idx(labels);
[sz1,sz2,~] = size(img1);
num_H = size(sam_H, 1); % number of superpixels
box_xy1 = zeros(num_H, 4);
sam_inv_H=zeros(num_H,9);
for i=1:num_H
    tmpsam=sam_idx{i};
    num_pixels=length(tmpsam);
    tmpH=reshape(sam_H(i,:),3,3);
    sam_inv_H(i,:)=reshape(tmpH\eye(3),1,9);
    [my, nx] = ind2sub([sz1,sz2], tmpsam');
    tmpmappts = tmpH*[nx; my; ones(1, num_pixels)];
    tmpmappts = tmpmappts(1:2,:)./repmat(tmpmappts(3,:),2,1);
    box_xy1(i, :)=[floor(min(tmpmappts(1,:))), floor(min(tmpmappts(2,:))), ceil(max(tmpmappts(1,:))), ceil(max(tmpmappts(2,:)))];
end


offset = [ 2 - min(1, min(box_xy1(:,1))) ; 2 - min(1, min(box_xy1(:,2)))];
cw = ceil(max([box_xy1(:,3); size(img1,2)]))+offset(1)-1;
ch = ceil(max([box_xy1(:,4); size(img1,1)]))+offset(2)-1;

if cw*ch>1e8
    panorama = img1;
    warped_img1=img1;
    warped_img2=img1;
    ssim_index=0;
    psnr_index=0;
    return;
end

%% backwarp texture mapping for each sagmented regions
num_in_h=max(labels(overlaped_C));
[warped_img1, mask1, warped_img2, mask2] = regionsBackwardMapping_matlab(img1, img2, labels, sam_inv_H, box_xy1, ch, cw, offset, num_in_h);
mask1_ = imfill(mask1,'holes');
holes = mask1_ & ~mask1;
warped_img1(cat(3,holes,holes,holes))=warped_img2(cat(3,holes,holes,holes));

final_mask1=imbinarize(rgb2gray(warped_img1),0);
[ssim_index, psnr_index]=full_reference_IQA(warped_img1, warped_img2);
panorama = imageBlending(warped_img1, final_mask1, warped_img2, mask2, 'linear');

end

