function [final_labels1 ,final_homos, overlapped_mask1] = sammaskLabeling(img1, img2, labels1, multi_homos, cell_matches)
% refine the masks generated via SAM by cross-homogrpahy minimal error comparison
% i.e. find the best fitted homography warp for each masked regions, index
% it with the index of homography
% anchor points are sampled in all the boundary of the non-overlapping region
    anchor_num=20;
    num_H=size(multi_homos,1);
    [sz1,sz2]=size(labels1);
    sam_idx1=label2idx(labels1);
    
    [X,Y]=meshgrid(1:sz2,1:sz1);
    C1=200;  C2=200;
    vega = 5;

    %% define the non-overlapping region for multi-homography warping
    full_matches1=[];
    full_matches2=[];
    for i=1:num_H
        tmpmatches1=cell_matches{i,1};
        tmpmatches2=cell_matches{i,2};
        full_matches1 = [full_matches1, tmpmatches1];
        full_matches2 = [full_matches2, tmpmatches2];
    end
    out_homo=calcHomo(full_matches1,full_matches2);
    map_XY=out_homo*[X(:),Y(:),ones(sz1*sz2,1)]';
    map_XY=map_XY(1:2,:)./repmat(map_XY(3,:),2,1);
    inner_index=map_XY(1,:)>=1 & map_XY(1,:)<=sz2 & map_XY(2,:)>=1 & map_XY(2,:)<=sz1;
    overlapped_mask1=false(sz1,sz2);
    inner_ind=sub2ind([sz1,sz2],Y(inner_index),X(inner_index));
    overlapped_mask1(inner_ind)=1;

    multi_S=cell(num_H, 1);
    theta=zeros(num_H, 1);
    for i=1:num_H
        tmpmatches1=cell_matches{i,1};
        tmpmatches2=cell_matches{i,2};
        S=calcSimilarity(tmpmatches1,tmpmatches2);
        multi_S{i}=S;
        theta(i)=atan(S(2)/S(1));
    end
    [~, index]=min(abs(theta));
    out_S=multi_S{index};

    %% first: homography fitting error calculation and minimal index indentification, 
    % applied on patial homographies where the region contains corresponding feature points
    refined_labels1=zeros(sz1,sz2);
    num_sam1=length(sam_idx1);
    for i=1:num_sam1
        tmpidx1=sam_idx1{i};
        if isempty(tmpidx1); continue; end
        mask_i=false(sz1,sz2);
        mask_i(tmpidx1)=1;
        tmpoverlap_mask_i=mask_i & overlapped_mask1;
        if sum(tmpoverlap_mask_i(:))>0 % if region lies entirely or partially in the overlapping reigon
            mask_i=tmpoverlap_mask_i;
            x=X(mask_i);
            y=Y(mask_i);
            valid_ind=[];
            for j=1:num_H
                tmpmatches1=cell_matches{j,1};
                ia=ismember(round(tmpmatches1'), [x,y], 'rows');
                if sum(ia)>0
                    valid_ind=[valid_ind,j];
                end
            end
            if isempty(valid_ind)
                valid_ind=(1:num_H);
            end
            if length(valid_ind)==1
                refined_labels1(mask_i)=valid_ind;
                continue;
            end
            mean_e=nan(num_H,1);
            for j=valid_ind
                tmpH=reshape(multi_homos(j,:),3,3);
                [color_error, in_perct]=calcPhotometricError(tmpH,[x,y],img1,img2);
                    mean_e(j)=mean(color_error)/in_perct;
            end
            [~, min_ind]=min(mean_e);
            refined_labels1(mask_i)=min_ind;
        end

    end 

    %% second: homography fitting for remaining pixels with label==0
    remaining_zeromask=refined_labels1==0 & overlapped_mask1;
    remaining_CC1=bwconncomp(remaining_zeromask);
    remaining_sam_idx_zero=remaining_CC1.PixelIdxList;
    for i=1:length(remaining_sam_idx_zero)
        tmpidx=remaining_sam_idx_zero{i};
        mask_i=false(sz1,sz2);
        mask_i(tmpidx)=1;
        dilate_mask_i=imdilate(mask_i,strel('square',3));
        bound_mask_i = dilate_mask_i & ~mask_i;
        valid_ind=refined_labels1(bound_mask_i);
        valid_ind(valid_ind==0)=[];
        if isempty(valid_ind); continue; end
        if length(valid_ind)==1
            refined_labels1(mask_i)=valid_ind;
        else
            N=histcounts(valid_ind,(0:max(valid_ind)+1));
            [~,N_ind]=max(N);
            refined_labels1(mask_i)=N_ind-1;
        end
    end
  
    %% third: homography fitting for non-overlapping region
    out_labels1=zeros(sz1,sz2);
    [cX,cY]=meshgrid(linspace(1,sz2,C2),linspace(1,sz1,C1));
    k=num_H+1;
    for i=1:size(cX,1)-1
        for j=1:size(cX,2)-1
            out_labels1(floor(cY(i,j)):floor(cY(i+1,j)),floor(cX(i,j)):floor(cX(i,j+1)))=k;
            k=k+1;
        end
    end
    out_labels1(refined_labels1~=0)=refined_labels1(refined_labels1~=0);

   %% Generate anchor points in the overlapping and non-overlapping boundary, 20 in each edge
    % first: polygon clipping
    contour_pts=contourExtract(out_homo, [size(img1,1),size(img1,2)]); % boundary edge of overlapping reigon
    ori_contour=[1 1; size(img1,2), 1; size(img1,2), size(img1,1); 1, size(img1,1); 1, 1];
    [in_edges_ori, out_edges_ori]=edgeComp(ori_contour, contour_pts);
    in_edges_ori=[max(1,min(sz2,in_edges_ori(:,1))), max(1,min(sz1,in_edges_ori(:,2))),max(1,min(sz2,in_edges_ori(:,3))), max(1,min(sz1,in_edges_ori(:,4)))];
    min_x=min(contour_pts(:,1));
    max_x=max(contour_pts(:,1));
    min_y=min(contour_pts(:,2));
    max_y=max(contour_pts(:,2));
    out_anchor_points=[];
    if (max_x-min_x)/sz2<0.6 || (max_y-min_y)/sz1<0.6  % if the diameter of the overlapping region is too small
        for i=1:size(out_edges_ori,1)
            pts_s=out_edges_ori(i,1:2);
            pts_e=out_edges_ori(i,3:4);
            dx=(pts_e(1)-pts_s(1))/(anchor_num-1);
            dy=(pts_e(2)-pts_s(2))/(anchor_num-1);
            out_anchor_points= [out_anchor_points; [pts_s(1)+dx.*(0:1:anchor_num-1)', pts_s(2)+dy.*(0:1:anchor_num-1)']];        
        end
        out_anchor_points = unique(out_anchor_points,'rows');
    end
    
    in_anchor_points = [];
    for i=1:size(in_edges_ori,1)
        pts_s=in_edges_ori(i,1:2);
        pts_e=in_edges_ori(i,3:4);
        dx=(pts_e(1)-pts_s(1))/(anchor_num-1);
        dy=(pts_e(2)-pts_s(2))/(anchor_num-1);
        in_anchor_points= [in_anchor_points; [pts_s(1)+dx.*(0:1:anchor_num-1)', pts_s(2)+dy.*(0:1:anchor_num-1)']];        
    end
    in_anchor_points = unique(in_anchor_points,'rows');
    
    
    anchor_mask=zeros(sz1,sz2);
    anchor_mask(sub2ind([sz1,sz2],round(in_anchor_points(:,2)),round(in_anchor_points(:,1))))=1;
    anchor_mask_=imdilate(anchor_mask,strel('square',5));
    neigh_anchor=[X(anchor_mask_ & overlapped_mask1), Y(anchor_mask_ & overlapped_mask1)];
    D=pdist2(in_anchor_points, neigh_anchor);
    [~, min_ind]=min(D,[],2);
    anchor_ind=sub2ind([sz1,sz2],neigh_anchor(min_ind,2),neigh_anchor(min_ind,1));
    anchor_labels=out_labels1(anchor_ind);
    
    %% calculate linearized homography for non-overlapping reigon
      % anchor_points linearization
      taylor_Ain=zeros(size(in_anchor_points,1),9);
      for i=1:size(in_anchor_points,1)
          label_i=anchor_labels(i);
          tmph=reshape(multi_homos(label_i,:),3,3);
          A = taylor_series(tmph, in_anchor_points(i,:));
          taylor_Ain(i,:)=A(:);
      end  

      taylor_Aout=[];
      if ~isempty(out_anchor_points)
          taylor_Aout=zeros(size(out_anchor_points,1),9);
          for i=1:size(out_anchor_points,1)
              A = taylor_series(out_S, out_anchor_points(i,:));
              taylor_Aout(i,:)=A(:);
          end
      end
      taylor_A=[taylor_Ain; taylor_Aout];

    out_idx=label2idx(out_labels1);
    num_labels=length(out_idx);
    final_homos=zeros(num_labels, 9);
    final_labels1=zeros(sz1,sz2);
    kk=1;
    for i=1:num_labels
        idx_i=out_idx{i};
        if isempty(idx_i); continue; end
        if i<=num_H
            final_labels1(idx_i)=kk;
            final_homos(kk,:)=multi_homos(i,:);
            kk=kk+1;
            continue;
        end
        pts_i=[X(idx_i), Y(idx_i)];
        mp_i=mean(pts_i,1);
        dist_mpi_ain=pdist2(mp_i, in_anchor_points);
        % Obtain kernel: Student's t-weighting 
        alphain = (1 + dist_mpi_ain./vega).^(-(vega+1)/2);
        if ~isempty(out_anchor_points)
            dist_mpi_aout=pdist2(mp_i, out_anchor_points);
            alphaout = (1 + dist_mpi_aout./vega).^(-(vega+1)/2);
        else
            alphaout=[];
        end

        alpha=[alphain,alphaout];
        alpha=alpha./sum(alpha);
        H_i=sum(repmat(alpha',1,9).*taylor_A, 1);
        final_homos(kk,:)=H_i./H_i(end);
        final_labels1(idx_i)=kk;
        kk=kk+1;
    end
    final_homos=final_homos(1:kk-1,:);

end


function [color_error, in_perct]=calcPhotometricError(H, pts, img1, img2)

[sz1,sz2,~]=size(img1);
w_pts=H*[pts'; ones(1,size(pts,1))];
w_pts=w_pts(1:2,:)./repmat(w_pts(3,:),2,1);
int_w_pts=floor(w_pts);
inner_ind=int_w_pts(1,:)>=1 & int_w_pts(1,:)<sz2 & int_w_pts(2,:)>=1 & int_w_pts(2,:)<sz1;
inner_wpts=w_pts(:,inner_ind);
ori_pts=pts(inner_ind',:)';
left_up_pts=[floor(inner_wpts(1,:)); floor(inner_wpts(2,:))];
left_down_pts=[floor(inner_wpts(1,:)); ceil(inner_wpts(2,:))];
right_up_pts=[ceil(inner_wpts(1,:)); floor(inner_wpts(2,:))];
right_down_pts=[ceil(inner_wpts(1,:)); ceil(inner_wpts(2,:))];
weights1 = (right_down_pts(1,:)-inner_wpts(1,:)).*(right_down_pts(2,:)-inner_wpts(2,:));
weights2 = (right_down_pts(1,:)-inner_wpts(1,:)).*(inner_wpts(2,:)-left_up_pts(2,:));
weights3 = (inner_wpts(1,:)-left_up_pts(1,:)).*(left_down_pts(2,:)-inner_wpts(2,:));
weights4 = (inner_wpts(1,:)-left_up_pts(1,:)).*(inner_wpts(2,:)-left_up_pts(2,:));
ind0=sub2ind([sz1,sz2],ori_pts(2,:),ori_pts(1,:));
ind1=sub2ind([sz1,sz2],left_up_pts(2,:),left_up_pts(1,:));
ind2=sub2ind([sz1,sz2],left_down_pts(2,:),left_down_pts(1,:));
ind3=sub2ind([sz1,sz2],right_up_pts(2,:),right_up_pts(1,:));
ind4=sub2ind([sz1,sz2],right_down_pts(2,:),right_down_pts(1,:));
r_value=weights1.*img2(ind1)+weights2.*img2(ind2)+weights3.*img2(ind3)+weights4.*img2(ind4);
g_value=weights1.*img2(ind1+sz1*sz2)+weights2.*img2(ind2+sz1*sz2)+weights3.*img2(ind3+sz1*sz2)+weights4.*img2(ind4+sz1*sz2);
b_value=weights1.*img2(ind1+2*sz1*sz2)+weights2.*img2(ind2+2*sz1*sz2)+weights3.*img2(ind3+2*sz1*sz2)+weights4.*img2(ind4+2*sz1*sz2);
color_error=sqrt(((img1(ind0)-r_value).^2+(img1(ind0+sz1*sz2)-g_value).^2+(img1(ind0+2*sz1*sz2)-b_value).^2)./3);
in_perct=sum(inner_ind)/size(int_w_pts,2);

end

function [contour_pts]=contourExtract(homo, sz)
% find the contour of the overlapping region
pts = [1, sz(2), sz(2), 1;
        1, 1, sz(1), sz(1);
        1, 1, 1, 1]; 
h_pt = homo*pts;
h_pt = h_pt(1:2,:)./repmat(h_pt(3,:),2,1);
h_boundpts=[h_pt'; h_pt(:,1)'];  % boundary edge for warped img1
tmppts=Sutherland_Hodgman(h_boundpts, sz);  % boundary edge for overlapping region
oripts=homo\[tmppts'; ones(1,size(tmppts,1))];
oripts=oripts(1:2,:)./repmat(oripts(3,:),2,1);
contour_pts=oripts';
end

function [ clip_pts] = Sutherland_Hodgman(map_pts, rec_sz)
% map_pts: the clipping polygon vertices 
% rec_sz: the rectangular window size
% designed only for convex polygon clipping

% per-window clipping
% first: up boundary
ind_map1=map_pts(:,2)<1;
tmppts1=[];
for i=1:size(map_pts,1)-1
    pts1=map_pts(i,:);
    pts2=map_pts(i+1,:);
    if ~ind_map1(i) && ind_map1(i+1)   % p1 in, p2 out
        intsect_py=1;
        intsect_px=(pts2(1)-pts1(1))*(1-pts1(2))/(pts2(2)-pts1(2))+pts1(1);
        tmppts1=[tmppts1; [intsect_px, intsect_py]];
        continue;
    end
    if ind_map1(i) && ~ind_map1(i+1)   % p1 out, p2 in
        intsect_py=1;
        intsect_px=(pts2(1)-pts1(1))*(1-pts1(2))/(pts2(2)-pts1(2))+pts1(1);
        tmppts1=[tmppts1; [intsect_px, intsect_py]; pts2];
        continue;
    end
    if ind_map1(i) && ind_map1(i+1)   % p1 out, p2 out
        continue;
    end
    if ~ind_map1(i) && ~ind_map1(i+1)   % p1 in, p2 in
        tmppts1=[tmppts1; pts2];
        continue;
    end 
end
tmppts1=[tmppts1; tmppts1(1,:)];

% second: right boundary
ind_map2=tmppts1(:,1)>rec_sz(2);
tmppts2=[];
for i=1:size(tmppts1,1)-1
    pts1=tmppts1(i,:);
    pts2=tmppts1(i+1,:);
    if ~ind_map2(i) && ind_map2(i+1)   % p1 in, p2 out
        intsect_px=rec_sz(2);
        intsect_py=(pts2(2)-pts1(2))*(intsect_px-pts1(1))/(pts2(1)-pts1(1))+pts1(2);
        tmppts2=[tmppts2; [intsect_px, intsect_py]];
        continue;
    end
    if ind_map2(i) && ~ind_map2(i+1)   % p1 out, p2 in
        intsect_px=rec_sz(2);
        intsect_py=(pts2(2)-pts1(2))*(intsect_px-pts1(1))/(pts2(1)-pts1(1))+pts1(2);
        tmppts2=[tmppts2; [intsect_px, intsect_py]; pts2];
        continue;
    end
    if ind_map2(i) && ind_map2(i+1)   % p1 out, p2 out
        continue;
    end
    if ~ind_map2(i) && ~ind_map2(i+1)   % p1 in, p2 in
        tmppts2=[tmppts2; pts2];
        continue;
    end 
end
tmppts2=[tmppts2; tmppts2(1,:)];


% third: bottom boundary
ind_map3=tmppts2(:,2)>rec_sz(1);
tmppts3=[];
for i=1:size(tmppts2,1)-1
    pts1=tmppts2(i,:);
    pts2=tmppts2(i+1,:);
    if ~ind_map3(i) && ind_map3(i+1)   % p1 in, p2 out
        intsect_py=rec_sz(1);
        intsect_px=(pts2(1)-pts1(1))*(intsect_py-pts1(2))/(pts2(2)-pts1(2))+pts1(1);
        tmppts3=[tmppts3; [intsect_px, intsect_py]];
        continue;
    end
    if ind_map3(i) && ~ind_map3(i+1)   % p1 out, p2 in
        intsect_py=rec_sz(1);
        intsect_px=(pts2(1)-pts1(1))*(intsect_py-pts1(2))/(pts2(2)-pts1(2))+pts1(1);
        tmppts3=[tmppts3; [intsect_px, intsect_py]; pts2];
        continue;
    end
    if ind_map3(i) && ind_map3(i+1)   % p1 out, p2 out
        continue;
    end
    if ~ind_map3(i) && ~ind_map3(i+1)   % p1 in, p2 in
        tmppts3=[tmppts3; pts2];
        continue;
    end 
end
tmppts3=[tmppts3; tmppts3(1,:)];


% fourth: left boundary
ind_map4=tmppts3(:,1)<1;
tmppts4=[];
for i=1:size(tmppts3,1)-1
    pts1=tmppts3(i,:);
    pts2=tmppts3(i+1,:);
    if ~ind_map4(i) && ind_map4(i+1)   % p1 in, p2 out
        intsect_px=1;
        intsect_py=(pts2(2)-pts1(2))*(intsect_px-pts1(1))/(pts2(1)-pts1(1))+pts1(2);
        tmppts4=[tmppts4; [intsect_px, intsect_py]];
        continue;
    end
    if ind_map4(i) && ~ind_map4(i+1)   % p1 out, p2 in
        intsect_px=1;
        intsect_py=(pts2(2)-pts1(2))*(intsect_px-pts1(1))/(pts2(1)-pts1(1))+pts1(2);
        tmppts4=[tmppts4; [intsect_px, intsect_py]; pts2];
        continue;
    end
    if ind_map4(i) && ind_map4(i+1)   % p1 out, p2 out
        continue;
    end
    if ~ind_map4(i) && ~ind_map4(i+1)   % p1 in, p2 in
        tmppts4=[tmppts4; pts2];
        continue;
    end 
end
tmppts4=[tmppts4; tmppts4(1,:)];

clip_pts = tmppts4;

end

function [in_edges, out_edges]=edgeComp(ori_vt, overlap_vt)
% compare edge in original edge and edge in the overlapping region
% exclude the overlapped parts and output rest edges
ori_edges=[ori_vt(1:end-1,:), ori_vt(2:end,:)];
overlap_edges=[overlap_vt(1:end-1,:), overlap_vt(2:end,:)];
dist_matrix=zeros(size(overlap_edges,1),size(ori_edges,1));
for i=1:size(overlap_edges,1)
    pts1=overlap_edges(i,1:2);
    pts2=overlap_edges(i,3:4);
    for j=1:size(ori_edges,1)
        x1=ori_edges(j,1);
        y1=ori_edges(j,2);
        x2=ori_edges(j,3);
        y2=ori_edges(j,4);
        a=y2-y1;
        b=x1-x2;
        c=x2*y1-x1*y2;
        dist_ij1=abs(a*pts1(1)+b*pts1(2)+c)/sqrt(a^2+b^2);
        dist_ij2=abs(a*pts2(1)+b*pts2(2)+c)/sqrt(a^2+b^2);
        dist_matrix(i,j)=(dist_ij1+dist_ij2)/2;
    end
end

min_dist1=min(dist_matrix,[],2);
min_dist2=min(dist_matrix,[],1);
in_edges=overlap_edges(min_dist1>=1e-1,:);
out_edges=ori_edges(min_dist2'>=1e-1,:);

end
