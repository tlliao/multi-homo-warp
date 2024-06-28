function [multi_homos, cell_matches]=multiHomoFitting(pts1,pts2, img1, img2, labels1, init_H, parameters)

% multi-model fitting method to find multiple subsets of feature points to
% estimate multiple homographies
% add segment anything model result to provide better smoothness
% cost 

fig_display=parameters.display;
para_dist=parameters.dist;
num_H=size(init_H,2);
colors_=rand(2*num_H,3);
[sz1,sz2,~]=size(img1);

%% the preprocess of PEARL
para_lambda=parameters.lambda; % The coefficient of smoothcost in energy function
para_beta=parameters.beta; % The coefficient of label cost in energy function
% para_max_label=100; % The maximum number of the initial models
para_maxdatacost=parameters.maxdatacost;
para_gamma=parameters.gamma; % distance between feature point and outlier model (datacost)

%% calculate the neighborhood system
h_pts1=pts1;
h_pts2=pts2;

num_sites = size(h_pts1,2); % number of matched SIFT feature points
neighbors = zeros(num_sites,num_sites); % Neiborhood system of match points

% Neighborhood system
tri_delaunay=delaunayTriangulation(pts1');  % Delaunay triangulation of feature points

% calculate the Neighbor Cost
for i=1:size(tri_delaunay,1)
   tri_1=tri_delaunay(i,1);
   tri_2=tri_delaunay(i,2);
   tri_3=tri_delaunay(i,3);
   
%    neighbors(min(tri_1,tri_2),max(tri_1,tri_2))=1;
%    neighbors(min(tri_1,tri_3),max(tri_1,tri_3))=1;
%    neighbors(min(tri_2,tri_3),max(tri_2,tri_3))=1; 
      
   l_1=labels1(round(pts1(2,tri_1)),round(pts1(1,tri_1)));
   l_2=labels1(round(pts1(2,tri_2)),round(pts1(1,tri_2)));
   l_3=labels1(round(pts1(2,tri_3)),round(pts1(1,tri_3)));
   if l_1==l_2
        neighbors(min(tri_1,tri_2),max(tri_1,tri_2))=1;
   end
   if l_1==l_3
        neighbors(min(tri_1,tri_3),max(tri_1,tri_3))=1;
   end
   if l_2==l_3
        neighbors(min(tri_2,tri_3),max(tri_2,tri_3))=1; 
   end
end


%% First Initial PEaRL
num_labels_0=num_H;

handle=GCO_Create(num_sites, num_labels_0+1); % include an outlier model

data_cost_0=int32(zeros(num_labels_0+1, num_sites)); 
smooth_cost_0=int32(para_lambda.*(ones(num_labels_0+1, num_labels_0+1)-eye(num_labels_0+1)));
label_cost_0=int32(para_beta*ones(1,num_labels_0+1));
% calculate the Data Cost of energy function

for id0=1:num_labels_0
    multi_H=init_H{id0};
    dist_error=calctransferError(multi_H, h_pts1, h_pts2);
    data_cost_0(id0,:)=min(dist_error,para_maxdatacost); % using the logistic filter
end
data_cost_0(num_labels_0+1,:)=para_gamma; % the outlier model's datacost


% run the PEaRL method
GCO_SetDataCost(handle, data_cost_0);
GCO_SetSmoothCost(handle, smooth_cost_0);
GCO_SetNeighbors(handle, neighbors);
GCO_SetLabelCost(handle,label_cost_0);
GCO_Expansion(handle);  

first_label = GCO_GetLabeling(handle); % the initial label 
[first_energy, ~, ~] = GCO_ComputeEnergy(handle);
GCO_Delete(handle);
multi_labels=first_label;

cell_sites_0=cell(1,num_labels_0+1); % the feature points' index in each model
num_sites_0=zeros(1,num_labels_0+1); % the number of points in each model
for i_1=1:num_labels_0+1
    tmp_sites=find(first_label==i_1); % the points in i-th model
    num_sites_0(i_1)=size(tmp_sites,1); % the number of points in i-th model
    cell_sites_0{i_1}=tmp_sites; % the points' index in i-th model
end
num_labels_1=sum(num_sites_0(1:end-1)>0); % the number of models of next iteration

% 对每个模型的特征点按特征点数目大小排序并在图像中显示出来
if fig_display
    [~,idx_Sites_0] = sort(num_sites_0,'descend');
    figure,imshow([img1,img2]);title(['first iteration with ' num2str(num_labels_0), ' labels']);
    hold on
    for i0=1:num_labels_0+1
        if idx_Sites_0(i0)==num_labels_0+1
            tmp_sites = cell_sites_0{idx_Sites_0(i0)};
            plot(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),'*','color',[0,0,0]); % 异常点颜色标为黑色
            plot(h_pts2(1,tmp_sites)+sz2,h_pts2(2,tmp_sites),'*','color',[0,0,0]);
        else
            tmp_sites = cell_sites_0{idx_Sites_0(i0)};
            plot(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),'*','color',colors_(i0,:)); % 在图像中标记处特征点发的分布
            text(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),num2str(i0));
            plot(h_pts2(1,tmp_sites)+sz2,h_pts2(2,tmp_sites),'*','color',colors_(i0,:)); % 在图像中标记处特征点发的分布
        end
    end
    hold off
end

% initial models of next iteration
cell_H_1 = cell(1,num_labels_1);
k_1=1;
for ih1=1:num_labels_0 
    if num_sites_0(ih1)>=4
        cell_H_1{k_1}=nonlinearSteOptimize(h_pts1(:,cell_sites_0{ih1}), h_pts2(:,cell_sites_0{ih1}));
        k_1=k_1+1;
    end
    if num_sites_0(ih1)>0 && num_sites_0(ih1)<4
        cell_H_1{k_1}=init_H{ih1};
        k_1=k_1+1;
    end
end


%% Second iteration of PEaRL

handle = GCO_Create(num_sites, num_labels_1+1);

data_cost_1 = int32(zeros(num_labels_1+1, num_sites)); % include outlier model
smooth_cost_1 = int32(para_lambda.*(ones(num_labels_1+1, num_labels_1+1)-eye(num_labels_1+1)));
label_cost_1 = int32(para_beta*ones(1,num_labels_1+1));
% calculate the Data Cost of energy function
for id1=1:num_labels_1
    multi_H=cell_H_1{id1};
    dist_error=calctransferError(multi_H, h_pts1, h_pts2);
    data_cost_1(id1,:)=min(dist_error,para_maxdatacost);
end
data_cost_1(num_labels_1+1,:)=para_gamma;

% run the PEaRL method
GCO_SetDataCost(handle, data_cost_1);
GCO_SetSmoothCost(handle, smooth_cost_1);
GCO_SetNeighbors(handle, neighbors);
GCO_SetLabelCost(handle,label_cost_1);
GCO_Expansion(handle);  

second_label = GCO_GetLabeling(handle);
[second_energy, ~, ~] = GCO_ComputeEnergy(handle);
GCO_Delete(handle);
if second_energy<=first_energy
    multi_labels=second_label;
end

cell_sites_1 = cell(1,num_labels_1+1);
num_sites_1 = zeros(1,num_labels_1+1); 
for i_2=1:num_labels_1+1
    tmp_sites = find(second_label==i_2);
    cell_sites_1{i_2} = tmp_sites;
    num_sites_1(i_2) = size(tmp_sites,1); 
end
num_labels_2=sum(num_sites_1(1:end-1)>0); % 第三次迭代初始模型数

% 对每个模型的特征点按点数大小排序并在图像中显示出来
if fig_display
    [~, idx_Sites_1] = sort(num_sites_1, 'descend');
    figure, imshow([img1,img2]);title(['2nd iteration with ' num2str(num_labels_1) ' labels']);
    hold on
    for i1=1:num_labels_1+1
        if idx_Sites_1(i1)==num_labels_1+1
            tmp_sites = cell_sites_1{idx_Sites_1(i1)};
            plot(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),'*','color',[0,0,0]); % 异常点颜色标为黑色
            plot(h_pts2(1,tmp_sites)+sz2,h_pts2(2,tmp_sites),'*','color',[0,0,0]);
        else
            tmp_sites = cell_sites_1{idx_Sites_1(i1)};
            plot(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),'*','color',colors_(i1,:));
            text(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),num2str(i1));
            plot(h_pts2(1,tmp_sites)+sz2,h_pts2(2,tmp_sites),'*','color',colors_(i1,:));
        end
    end
    hold off
end

% initial models of third iteration
cell_H_2 = cell(1,num_labels_2);
k_2=1;
for ih2=1:num_labels_1 
    if num_sites_1(ih2)>=4
        cell_H_2{k_2}=nonlinearSteOptimize(h_pts1(:,cell_sites_1{ih2}), h_pts2(:,cell_sites_1{ih2}));
        k_2=k_2+1;
    end
    if num_sites_1(ih2)>0 && num_sites_1(ih2)<4
        cell_H_2{k_2} = cell_H_1{ih2};
        k_2=k_2+1;
    end
end

%% Iterating Run the PEaRL Method, if the energy isn't decreasing, stop
iteration_number = 2;
while second_energy<first_energy
    first_energy = second_energy;
    handle = GCO_Create(num_sites, num_labels_2+1);
    
    data_cost_2 = int32(zeros(num_labels_2+1, num_sites)); % include an outlier model
    smooth_cost_2 = int32(para_lambda.*(ones(num_labels_2+1, num_labels_2+1)-eye(num_labels_2+1)));
    label_cost_2 = int32(para_beta*ones(1,num_labels_2+1));
    % calculate the Data Cost of energy function
    for id2=1:num_labels_2
        multi_H = cell_H_2{id2};
        dist_error=calctransferError(multi_H, h_pts1, h_pts2);
        data_cost_2(id2,:)=min(dist_error,para_maxdatacost);
    end
    data_cost_2(num_labels_2+1,:)=para_gamma;
    %toc
    
    % run the PEaRL method
    GCO_SetDataCost(handle, data_cost_2);
    GCO_SetSmoothCost(handle, smooth_cost_2);
    GCO_SetNeighbors(handle, neighbors);
    GCO_SetLabelCost(handle,label_cost_2);
    GCO_Expansion(handle);  
    
    third_label=GCO_GetLabeling(handle);
    [third_energy, ~, ~]=GCO_ComputeEnergy(handle);
    GCO_Delete(handle);
    multi_labels=third_label;
    
    iteration_number=iteration_number + 1;

    cell_Sites_2=cell(1,num_labels_2+1);
    num_Sites_2=zeros(1,num_labels_2+1);
    for i_3=1:num_labels_2+1
        tmp_sites = find(third_label==i_3);
        cell_Sites_2{i_3} = tmp_sites;
        num_Sites_2(i_3) = size(tmp_sites,1);
    end
    num_labels_3=sum(num_Sites_2(1:end-1)>0);% 下一次迭代的模型数
    
    if fig_display
        [~,idx_Sites_2] = sort(num_Sites_2, 'descend'); 
        % 对每个模型的特征点按点数大小排序并在图像中显示出来
        figure, imshow([img1,img2]);
        fig_str_2 = num2str(iteration_number);
        title([fig_str_2 'st iteration with ' num2str(num_labels_2) ' labels']);
        hold on
        for i2=1:num_labels_2+1
            if idx_Sites_2(i2)==num_labels_2+1
                tmp_sites = cell_Sites_2{idx_Sites_2(i2)};
                plot(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),'*','color',[0,0,0]); % 异常点颜色标为黑色
                plot(h_pts2(1,tmp_sites)+sz2,h_pts2(2,tmp_sites),'*','color',[0,0,0]);
            else
                tmp_sites = cell_Sites_2{idx_Sites_2(i2)};
                plot(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),'*','color',colors_(i2,:));
                text(h_pts1(1,tmp_sites),h_pts1(2,tmp_sites),num2str(i2));
                plot(h_pts2(1,tmp_sites)+sz2,h_pts2(2,tmp_sites),'*','color',colors_(i2,:));
            end
        end
        hold off
    end
    
    % initial models of next iteration
    cell_H_3 = cell(1,num_labels_3);
    k_3=1;
    for ih3=1:num_labels_2 
        if num_Sites_2(ih3)>=4
            cell_H_3{k_3}=nonlinearSteOptimize(h_pts1(:,cell_Sites_2{ih3}), h_pts2(:,cell_Sites_2{ih3}));
            k_3=k_3+1;
        end
        if num_Sites_2(ih3)>0 && num_Sites_2(ih3)<4
            cell_H_3{k_3} = cell_H_2{ih3};
            k_3=k_3+1;
        end
    end
        
    num_labels_2=num_labels_3;
    cell_H_2=cell_H_3;  
    second_energy=third_energy;
end


num_labels=max(multi_labels);
cell_sites=cell(1,num_labels); % the feature points' index in each model
num_pts=zeros(1,num_labels); % the number of points in each model
for tmpi=1:num_labels
    tmp_sites = find(multi_labels==tmpi); % the points in i-th model
    num_pts(tmpi) = size(tmp_sites,1); % the number of points in i-th model
    cell_sites{tmpi} = tmp_sites; % the points' index in i-th model
end
ind=1:num_labels;
valid_ind=ind(num_pts>=4);  % model with matches bigger than 20 is useful
valid_ind(valid_ind==num_labels)=[];  % num_labels corresponds the outlier model
num_hypothesis=length(valid_ind);  
cell_matches=cell(num_hypothesis,2);
multi_homos=zeros(num_hypothesis,9);
k=1;
for tmpi=1:num_hypothesis
        tmp_ind=cell_sites{valid_ind(tmpi)};
        tmpmatches1=pts1(:,tmp_ind);
        tmpmatches2=pts2(:,tmp_ind);
        tmpH=nonlinearSteOptimize(tmpmatches1, tmpmatches2);
        dist_i=calctransferError(tmpH, tmpmatches1, tmpmatches2);
        inner_ind=dist_i<=para_dist;
        if sum(inner_ind)<4; continue; end
        tmpH_=nonlinearSteOptimize(tmpmatches1(:,inner_ind), tmpmatches2(:,inner_ind));
        i_flag=homoValidJudge(tmpH_,[sz1,sz2]);
        if i_flag
            multi_homos(k,:)=tmpH_(:);
            cell_matches{k,1}=tmpmatches1(:,inner_ind);
            cell_matches{k,2}=tmpmatches2(:,inner_ind);
            k=k+1;
        end
        
end
cell_matches=cell_matches(1:k-1,:);
multi_homos=multi_homos(1:k-1,:);


end