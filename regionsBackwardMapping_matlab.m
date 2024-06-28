function [warped_img1, warped_mask1, warped_img2, warped_mask2]=regionsBackwardMapping_matlab(img1, img2, labels, invH, box_xy, ch, cw, offset, num_in_h)
    %% blendering warped target image based on multiple homographies fitted for multiple segmented regions
    % initialization
    num_h = size(invH,1);
    [sz1,sz2,~] = size(img1);
    IND=(1:ch*cw)';
    cellwarped_imgs=cell(num_in_h+1,2);
    warped_img1=zeros(ch,cw,3);
    warped_img2=zeros(ch,cw,3);
    warped_mask1=zeros(ch,cw);
    warped_mask2=zeros(ch,cw);
    
    % blender warped reference image
    warped_img2(offset(2):(offset(2)+size(img2,1)-1),offset(1):(offset(1)+size(img2,2)-1),:) = img2;
    warped_mask2(offset(2):(offset(2)+size(img2,1)-1),offset(1):(offset(1)+size(img2,2)-1)) = 1;
    
    % blender the overlapping region of warped target image
    for i=1:num_in_h
        in_imgi=zeros(ch,cw,3);
        in_maski=zeros(ch,cw);
        mleft  = box_xy(i);
        mup  = box_xy(i+num_h);
        mright  = box_xy(i+2*num_h);
        mdown  = box_xy(i+3*num_h);   
        for x=mleft:mright
            for y=mup:mdown
                posx = (invH(num_h*0+i)*x+invH(num_h*3+i)*y+invH(num_h*6+i))/(invH(num_h*2+i)*x+invH(num_h*5+i)*y+invH(num_h*8+i));
                posy = (invH(num_h*1+i)*x+invH(num_h*4+i)*y+invH(num_h*7+i))/(invH(num_h*2+i)*x+invH(num_h*5+i)*y+invH(num_h*8+i));
                px = floor(posx);
                py = floor(posy);
                if (px>=1 && px<sz2 && py>=1 && py<sz1)
                    l1=labels(py,px);
                    l2=labels(py+1,px);
                    l3=labels(py,px+1);
                    l4=labels(py+1,px+1);
                    if(l1==i || l2==i || l3==i || l4==i)
                        map_x=x+offset(1)-1;
                        map_y=y+offset(2)-1;
                        if (map_x>=1 && map_x<=cw && map_y>=1 && map_y<=ch)
                            in_imgi(map_y,map_x,1)=(px+1-posx)*(py+1-posy)*img1(py,px,1) + (px+1-posx)*(posy-py).*img1(py+1,px,1) + (posx-px)*(py+1-posy)*img1(py,px+1,1) + (posx-px)*(posy-py)*img1(py+1,px+1,1);
                            in_imgi(map_y,map_x,2)=(px+1-posx)*(py+1-posy)*img1(py,px,2) + (px+1-posx)*(posy-py).*img1(py+1,px,2) + (posx-px)*(py+1-posy)*img1(py,px+1,2) + (posx-px)*(posy-py)*img1(py+1,px+1,2);
                            in_imgi(map_y,map_x,3)=(px+1-posx)*(py+1-posy)*img1(py,px,3) + (px+1-posx)*(posy-py).*img1(py+1,px,3) + (posx-px)*(py+1-posy)*img1(py,px+1,3) + (posx-px)*(posy-py)*img1(py+1,px+1,3);
                            in_maski(map_y,map_x)=1;
                        end
                    end
                end
            end
        end
        cellwarped_imgs{i,1}=in_imgi;
        cellwarped_imgs{i,2}=in_maski;
    end
    
    % compare each blendered parts in the overlapping region to deal with
    % duplicate issues
    for i=1:num_in_h-1
        for j=i+1:num_in_h
            tmpimgi=cellwarped_imgs{i,1};
            tmpmaski=cellwarped_imgs{i,2};
            tmpimgj=cellwarped_imgs{j,1};
            tmpmaskj=cellwarped_imgs{j,2};
            overlap_maskij=tmpmaski & tmpmaskj;
            overlap_indij=IND(overlap_maskij);
            if ~isempty(overlap_indij)   % if duplicate issue exist
                tmperror_i=[tmpimgi(overlap_indij)-warped_img2(overlap_indij), tmpimgi(overlap_indij+ch*cw)-warped_img2(overlap_indij+ch*cw),tmpimgi(overlap_indij+2*ch*cw)-warped_img2(overlap_indij+2*ch*cw)];
                tmperror_j=[tmpimgj(overlap_indij)-warped_img2(overlap_indij), tmpimgj(overlap_indij+ch*cw)-warped_img2(overlap_indij+ch*cw),tmpimgj(overlap_indij+2*ch*cw)-warped_img2(overlap_indij+2*ch*cw)];
                mean_error_i=mean(sqrt(sum(tmperror_i.^2, 2)./3));
                mean_error_j=mean(sqrt(sum(tmperror_j.^2, 2)./3));
                if mean_error_j<mean_error_i
                    tmpimgi(overlap_indij)=0;
                    tmpimgi(overlap_indij+ch*cw)=0;
                    tmpimgi(overlap_indij+2*ch*cw)=0;
                    tmpmaski(overlap_indij)=0;
                else
                    tmpimgj(overlap_indij)=0;
                    tmpimgj(overlap_indij+ch*cw)=0;
                    tmpimgj(overlap_indij+2*ch*cw)=0;
                    tmpmaskj(overlap_maskij)=0;
                end
            end
            cellwarped_imgs{i,1}=tmpimgi;
            cellwarped_imgs{i,2}=tmpmaski;
            cellwarped_imgs{j,1}=tmpimgj;
            cellwarped_imgs{j,2}=tmpmaskj;
        end
    end
    
    % blender the non-overlapping region of warped target image
    out_imgi=zeros(ch,cw,3);
    out_maski=zeros(ch,cw);
    for i=num_in_h+1:num_h
        mleft  = box_xy(i);
        mup  = box_xy(i+num_h);
        mright  = box_xy(i+2*num_h);
        mdown  = box_xy(i+3*num_h);   
        for x=mleft:mright
            for y=mup:mdown
                posx = (invH(num_h*0+i)*x+invH(num_h*3+i)*y+invH(num_h*6+i))/(invH(num_h*2+i)*x+invH(num_h*5+i)*y+invH(num_h*8+i));
                posy = (invH(num_h*1+i)*x+invH(num_h*4+i)*y+invH(num_h*7+i))/(invH(num_h*2+i)*x+invH(num_h*5+i)*y+invH(num_h*8+i));
                px = floor(posx);
                py = floor(posy);
                if (px>=1 && px<sz2 && py>=1 && py<sz1)
                    l1=labels(py,px);
                    l2=labels(py+1,px);
                    l3=labels(py,px+1);
                    l4=labels(py+1,px+1);
                    if(l1==i || l2==i || l3==i || l4==i)
                        map_x=x+offset(1)-1;
                        map_y=y+offset(2)-1;
                        if (map_x>=1 && map_x<=cw && map_y>=1 && map_y<=ch)
                            out_imgi(map_y,map_x,1)=(px+1-posx)*(py+1-posy)*img1(py,px,1) + (px+1-posx)*(posy-py).*img1(py+1,px,1) + (posx-px)*(py+1-posy)*img1(py,px+1,1) + (posx-px)*(posy-py)*img1(py+1,px+1,1);
                            out_imgi(map_y,map_x,2)=(px+1-posx)*(py+1-posy)*img1(py,px,2) + (px+1-posx)*(posy-py).*img1(py+1,px,2) + (posx-px)*(py+1-posy)*img1(py,px+1,2) + (posx-px)*(posy-py)*img1(py+1,px+1,2);
                            out_imgi(map_y,map_x,3)=(px+1-posx)*(py+1-posy)*img1(py,px,3) + (px+1-posx)*(posy-py).*img1(py+1,px,3) + (posx-px)*(py+1-posy)*img1(py,px+1,3) + (posx-px)*(posy-py)*img1(py+1,px+1,3);
                            out_maski(map_y,map_x)=1;
                        end
                    end
                end
            end
        end
    end
    cellwarped_imgs{num_in_h+1,1}=out_imgi;
    cellwarped_imgs{num_in_h+1,2}=out_maski;
    
    % combine all warped regions into warped target image
    for i=1:num_in_h+1
        warped_img1=warped_img1+cellwarped_imgs{i,1};
        warped_mask1 = warped_mask1 + cellwarped_imgs{i,2};
    end
    warped_img1=warped_img1./cat(3,warped_mask1,warped_mask1,warped_mask1);
    warped_img1(isnan(warped_img1(:)))=0;
    warped_mask1(warped_mask1>0)=1;
        
    
%--- function end, return    
end
