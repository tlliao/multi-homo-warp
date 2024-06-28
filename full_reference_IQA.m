function [ssim_error, psnr_score] = full_reference_IQA(target, reference)
% calculate the ssim error between img and ref,
% only the overlapping region is taken into account
    if isa(target,'double')
        target=uint8(target.*255);
        reference=uint8(reference.*255);
    end

    %% ssim quality
    mask_tar = imbinarize(rgb2gray(target),0);
    mask_ref = imbinarize(rgb2gray(reference),0);
    mask_C = uint8(mask_tar & mask_ref);
    overlap_ref = reference.*cat(3,mask_C,mask_C,mask_C);
    overlap_tar = target.*cat(3,mask_C,mask_C,mask_C);
    [m, n] = ind2sub(size(mask_C),find(mask_C));
    min_m = min(m);
    max_m = max(m);
    min_n = min(n);
    max_n = max(n);
    
    overlap_tar_ = overlap_tar(min_m:max_m, min_n:max_n,:);
    overlap_ref_ = overlap_ref(min_m:max_m, min_n:max_n,:);
    ssim_error = ssim(overlap_tar_, overlap_ref_);

    %% Peak signal-to-noise ratio (PSNR) quality
    psnr_score = psnr(overlap_tar_, overlap_ref_);  

    if isempty(ssim_error)
        ssim_error=0;
    end
    if isempty(psnr_score)
        psnr_score=0;
    end
    
end

