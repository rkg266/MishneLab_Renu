function [L_all, mergedA_all] = runROI_meso_nlm_new_v1(cfg, PixxTime_dff, allregionspix, allregions)
if ~isfield(cfg, 'min_filled_area')
    cfg.min_filled_area = 0.97;
end
if ~isfield(cfg,'bordercrop')
    cfg.bordercrop=0;
end
if ~isfield(cfg,'N_TRIALS')
    cfg.N_TRIALS=1;
end
if ~isfield(cfg,'makePlots')
    cfg.makePlots=0;
end
if ~isfield(cfg,'N_EIG')
    cfg.N_EIG = 51;
end
if ~isfield(cfg,'preProcess')
    cfg.preProcess=1;
end
if ~isfield(cfg,'n_clust')
    cfg.n_clust=100;
end
if ~isfield(cfg,'refine')
    cfg.refine=1;
end
if ~isfield(cfg,'doSizeThresh')
    cfg.SizeThresh=[30,20000];
end
if ~isfield(cfg,'normalize')
    cfg.normalize=1;
end
if ~isfield(cfg,'pca')
    cfg.pca=1;
end
if ~isfield(cfg,'mxcluster')
    cfg.mxcluster = 1000;
end
if ~isfield(cfg,'thrcluster')
    cfg.thrcluster = 0.95;
end
if ~isfield(cfg,'pca_start_ind')
    cfg.pca_start_ind = 1;
end
if ~isfield(cfg,'isoverlap')
    cfg.isoverlap = false;
end
if ~isfield(cfg, 'nomissingpixels')
    cfg.nomissingpixels = true;
end
% @renu
if ~isfield(cfg, 'ComputeTemporalCorr')
    cfg.ComputeTemporalCorr = false;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
cfg.fig_str = [cfg.title_str,'_preproc_',num2str(cfg.preProcess),'_normal_',num2str(cfg.normalize),...
    '_pca_',num2str(cfg.pca),'_ntrial_',num2str(cfg.N_TRIALS),'_neig_',num2str(cfg.N_EIG)];
cfg.fig_str

NROWS=cfg.NROWS;
NCOLS=cfg.NCOLS;
min_filled_area = cfg.min_filled_area;

A_full=zeros(NROWS*NCOLS,size(PixxTime_dff,2));
A_full(allregionspix,:)=PixxTime_dff;
A_full=reshape(A_full,NROWS,NCOLS,[]);
if cfg.preProcess
    A_full=preprocess_data(A_full);  % this was no where defined
end
A_full(isnan(A_full)|isinf(A_full))=0;

cfg.N_TIME = floor(size(A_full,3) / cfg.N_TRIALS);
A_full=A_full(:,:,1:cfg.N_TIME* cfg.N_TRIALS);
BW=allregions;
%
Lambda = zeros(cfg.N_TRIALS,cfg.N_EIG-1);
embedding = struct('Psi', repmat({zeros(cfg.NROWS*cfg.NCOLS,cfg.N_EIG-1)}, cfg.N_TRIALS, 1));
embedding_norm = zeros(cfg.NROWS,cfg.NCOLS,cfg.N_TRIALS);

mergedA_all = cell(length(cfg.thrcluster), 1); % renu
t1=toc;

%%
tic;
for trial_ind = 1:cfg.N_TRIALS
    %
    configParams = [];
    configParams.self_tune = 4; %32; % @renu 
    configParams.kNN = 16; %50; % @renu 
    configParams.normalization = 'markov';
    configParams.maxInd = cfg.N_EIG;
    configParams.doNormalize = cfg.normalize;
    configParams.doPCA = cfg.pca;
    
    time_inds = (1:cfg.N_TIME) + (trial_ind-1)*cfg.N_TIME;
    disp(trial_ind)
    
    X = (A_full(:,:,time_inds));
    if configParams.doNormalize
        mX = mean(X,3);
        X = bsxfun(@minus, X, mX);
        sX = sqrt(mean(X.*X, 3)) + eps;
        X = bsxfun(@times, X, 1./sX);
        X = medfilt3(X,[3 3 3]);
        tmp = reshape(X,[],cfg.N_TIME);
        tmp = tmp(BW(:)>0,:);
        %tmp=reshape(tmp,NROWS,NCOLS,[]);
    else
        tmp = reshape(X,[],cfg.N_TIME);
        tmp = tmp(BW(:)>0,:);
        %tmp=reshape(tmp,NROWS,NCOLS,[]);
    end
    %X = preprocess_data(X);
    if configParams.doPCA
        tmp = bsxfun(@minus,tmp,mean(tmp));
        tmp(isnan(tmp))=0;
        [U,S,~] = pcafast(tmp,100);
        [~,thresh_ind] = elbow_in_graph(fliplr(log(diag(S))),false);
        stind = cfg.pca_start_ind;
        tmp = U(:,stind:thresh_ind)*S(stind:thresh_ind,stind:thresh_ind);
    end
    %
    K{trial_ind} = calcAffinityMat(tmp', configParams);
    
    [~, lambda, Psi, ~, Phi] = calcDiffusionMap(K{trial_ind},configParams);
    figure;plot(lambda);
    Lambda(trial_ind,:) = lambda(2:end);
    %Psi = Psi./ repmat(sqrt(sum(Psi.^2)),size(Psi,1),1);
    %Phi = Phi./ repmat(sqrt(sum(Phi.^2)),size(Phi,1),1);
    
    if 0   % plotting eigen vectors (make if 1 for eigen analysis)
        figure;
        %ha = tight_subplot(2,4,[.1 .01],[.1 .05],[.02 .02]); @renu
        set(gcf,'Position',[129 193 1347 785])
        eig_inds = 2:51;
        
        for i = 0:min(47,size(Psi,2)-2)
            temp = zeros(NROWS*NCOLS,1);
            temp(BW(:)>0) = Psi(:,eig_inds(i+1));
            temp=reshape(temp,NROWS,NCOLS);
            %axes(ha(mod(i,8)+1)); @renu
            subplot(2, 4, mod(i,8)+1); % @renu
            imagesc(temp);axis image; colorbar;
            title(['\Psi_{' num2str(i+1) '}'])
            if mod(i+1,8)==0 && cfg.makePlots
                saveas(gcf,sprintf('%s_trial_%d_psi_%d.png',fullfile(cfg.outputfilepath,cfg.fig_str)...
                    ,trial_ind,i+1));
            end
        end
    end
    
    temp = zeros(NROWS,NCOLS);
    temp(BW(:)>0) = sqrt(sum(Psi.^2,2));
    
    temp3d = zeros(NROWS,NCOLS,3);
    temp3d(:,:,2) = temp;
    
    figure;
    imagesc(temp3d* 1/quantile(temp(:),0.99));axis image;title('embedding norm')
    if cfg.makePlots
        saveas(gcf,sprintf('%s_trial_%d_embedding_norm.fig',fullfile(cfg.outputfilepath,cfg.fig_str),trial_ind));
    end
    embedding(trial_ind).Psi(BW(:)>0,:) = Psi(:,2:end);
    embedding_norm(:,:,trial_ind) = temp;
    
end
close all
%
embedding_all = [embedding.Psi];
t2=toc;

%%
%%%%%%%%%%%%%%%% Cluster regions
tic;
% refine using elbow in graph
n_eig = cfg.N_EIG-1;
tmp = reshape(A_full,[],cfg.N_TIME*cfg.N_TRIALS);
remove_inds=[];

embedding_norm_part =  zeros(NROWS,NCOLS,cfg.N_TRIALS);
for trial_ind = 1:cfg.N_TRIALS
    temp = zeros(NROWS,NCOLS);
    temp(:) = sqrt(sum(embedding(trial_ind).Psi(:,1:n_eig).^2,2));
    embedding_norm_part(:,:,trial_ind) = temp;
end
% %%
max_Psi = max(embedding_norm_part,[],3);
max_Psi(remove_inds) = nan;
temp3d=zeros(NROWS,NCOLS,3);
temp3d(:,:,2) = max_Psi;
if cfg.makePlots
    figure;imagesc(temp3d* 1/quantile(max_Psi(:),0.99) );axis image
end
for iclust=1:length(cfg.n_clust)
    if ~exist('max_inds','var')
        [~,max_inds] = sort(max_Psi(:),'descend');
        max_inds(isnan(max_Psi(max_inds))) = [];
    end
    %if 0
    max_inds = find(imregionalmax(imgaussfilt(max_Psi)));
    [~,I] = sort(max_Psi(max_inds),'descend');
    max_inds = max_inds(I);
    rem_inds = setdiff(1:length(max_Psi(:)),max_inds)';
    [~,J] = sort(max_Psi(rem_inds),'descend');
    max_inds = [max_inds;rem_inds(J)];
    % end
    
    clust_ind = 1;
    temp=nan(NROWS,NCOLS);
    clust = false(NROWS*NCOLS,cfg.n_clust(iclust));
    clust0 = false(NROWS*NCOLS,cfg.n_clust(iclust));
    embedding_all = [embedding.Psi];
    %
    while sum(sum(clust,2)>0)/length(allregionspix) < min_filled_area  && ~isempty(max_inds) %&& clust_ind <= cfg.n_clust(iclust) max_Psi(max_inds(1))>0.05 %length(max_inds)
        pix_ind = max_inds(1);
        %     tempPsi = [];
        %     for j = 1:N_TRIALS
        %         Psi = embedding(j).Psi;
        %         [~,sortPsi] =sort(abs(Psi(pix_ind,:)),'descend');
        %         tempPsi = [tempPsi, Psi(:,sortPsi(1:3))];
        %     end
        s = (embedding_all(pix_ind,:) ).^2;%abs(embedding_all(pix_ind,:) ); %
        tempPsi = embedding_all(:,(s/max(s)>0.1));
        normal = tempPsi(pix_ind,:) ./ norm(tempPsi(pix_ind,:));
        proj = normal*tempPsi';
        %     [~,max_ind] = max(proj);
        max_ind = pix_ind;
        
        D1=pdist2(tempPsi,tempPsi(pix_ind,:));
        D2=pdist2(tempPsi,zeros(size(tempPsi(pix_ind,:))));
        dist_inds = D1 < D2;
        
        if sum(dist_inds(:)) > 10
            
            temp2 = zeros(NROWS,NCOLS);
            temp2(dist_inds) = 1;
            
            CC = bwconncomp(temp2>0);
            idx = (cellfun(@(x)intersect(x,max_ind),CC.PixelIdxList,'UniformOutput',false));
            idx = find(cellfun(@any,idx));
            dist_inds = CC.PixelIdxList{idx};
            
            if length(dist_inds) < 10
                max_inds(1) = nan;
                max_inds(isnan(max_inds)) = [];
                [~,sortedInds] = sort(max_Psi(max_inds),'descend');
                max_inds = max_inds(sortedInds);
                continue
            end
            
            if cfg.makePlots
                [r c] = ind2sub([NROWS,NCOLS],pix_ind);
                temp2 = zeros(NROWS,NCOLS);
                temp2(:) = D1-D2;
                figure(200);
                clf
                subplot(221);
                imagesc(temp2);axis image
                hold on
                scatter(c,r,'k','filled')
                subplot(222);
                temp2 = zeros(NROWS,NCOLS);
                temp2(dist_inds) = 1;
                imagesc(temp2);axis image
                hold on
                scatter(c,r,'k','filled')
            end
            clust0(dist_inds,clust_ind) = 1;
            
            if cfg.refine
                s = (sum(embedding_all(dist_inds,:) ).^2 );
                %s = abs(sum(embedding_all(dist_inds,:) ));
                tempPsi = embedding_all(:,(s/max(s)>0.1));
                %[~,max_ind] = max(sum(tempPsi(dist_inds,:).^2,2 ));
                %max_ind = dist_inds(max_ind);
                %D1=pdist2(tempPsi,(tempPsi(max_ind,:)));
                [~,max_ind] = max(sum(tempPsi(dist_inds,:).^2,2));
                max_ind = dist_inds(max_ind);
                D1=pdist2(tempPsi,(tempPsi(max_ind,:)));
                [D1_sorted,sorted_inds]=sort(D1);
                [thresh,thresh_ind] = elbow_in_graph(D1_sorted);
                dist_inds = D1 < 0.9*thresh;
                
                temp2 = zeros(NROWS,NCOLS);
                temp2(dist_inds) = 1;
                %[~,max_ind] = min(D1);
                [r c] = ind2sub([NROWS,NCOLS],max_ind);
                CC = bwconncomp(imfill(temp2>0,'holes'));
                idx = (cellfun(@(x)intersect(x,max_ind),CC.PixelIdxList,'UniformOutput',false));
                idx = find(cellfun(@any,idx));
                dist_inds = CC.PixelIdxList{idx};
                
                if cfg.makePlots
                    temp2 = zeros(NROWS,NCOLS);
                    temp2(:) = D1;
                    figure(200);
                    subplot(223);
                    imagesc(temp2);axis image
                    hold on
                    scatter(c,r,'k','filled')
                    
                    subplot(224);
                    temp2 = zeros(NROWS,NCOLS);
                    temp2(dist_inds) = 1;
                    imagesc(temp2);axis image
                    hold on
                    scatter(c,r,'k','filled')
                    title(num2str(clust_ind))
                    drawnow
                end
            end
            
            max_inds(1) = nan;%max_inds(max_rem_ind);
            for j = 1:length(dist_inds)
                max_inds(max_inds == dist_inds(j)) = nan;
            end
            max_inds(isnan(max_inds)) = [];
            
            ecc = regionprops(CC,'Eccentricity');
            if ecc(idx).Eccentricity >0.95
                continue;
            end
            
            temp(dist_inds) = min(clust_ind,temp(dist_inds));
            
            clust(dist_inds,clust_ind) = 1;
            clust_ind = clust_ind +1;
            
            %[~,sortedInds] = sort(D1(max_inds),'descend');
            %max_inds = max_inds(sortedInds);
            
            tmp2d=zeros(size(max_Psi));
            tmp2d(max_inds) = max_Psi(max_inds);
            %figure(120);imagesc(tmp2d);axis image;drawnow
        else
            max_inds(1) = nan;
            %         dist_inds = find(dist_inds);
            %         for j = 1:length(dist_inds)
            %             max_inds(max_inds == dist_inds(j)) = nan;
            %         end
            max_inds(isnan(max_inds)) = [];
            [~,sortedInds] = sort(max_Psi(max_inds),'descend');
            max_inds = max_inds(sortedInds);
        end
    end
    %squ
    clust = clust(:,1:clust_ind-1);
    n_clust = size(clust,2);
    %temp(isnan(temp(:))) = 0;
    %props = regionprops(temp,'Area','Solidity','Extent','BoundingBox');
    
    tmp = reshape(A_full,NROWS*NCOLS,[]);
    ROI = zeros(n_clust,cfg.N_TIME*cfg.N_TRIALS);
    for k = 1:n_clust
        ROI(k,:) = mean(tmp(clust(:,k),:));
    end
    if cfg.makePlots
        figure;imagesc(ROI);
    end
    t3=toc;
    %%
    %%%%%%%%%%%%%%% merge_clusters
    tic;
    A = double(clust);
    nr = size(A,2);
    
    A_corr = triu(A(:,1:nr)'*A(:,1:nr));
    A_corr(1:nr+1:nr^2) = 0;
    FF2 = A_corr > 0;
    
    C = ROI;
    C_corr = zeros(nr);
    for i = 1:nr
        overlap_indeces = find(A_corr(i,:));
        if ~isempty(overlap_indeces)
            corr_values = corr(C(i,:)',C(overlap_indeces,:)');
            C_corr(i,overlap_indeces) = corr_values;
            C_corr(overlap_indeces,i) = corr_values;
        end
    end
    for ithrcluster=1:length(cfg.thrcluster)
        FF1 = triu(C_corr)>= cfg.thrcluster(ithrcluster);
        FF3 = and(FF1,FF2);
        %[l,c] = graph_conn_comp_mex(sparse(FF3+FF3'));     % extract connected components
        g = graph(sparse(FF3+FF3'));
        l = conncomp(g);
        c = length(unique(l));
        
        MC = [];
        for i = 1:c
            if length(find(l==i))>1
                MC = [MC,(l==i)'];
            end
        end
        %
        cor = zeros(size(MC,2),1);
        for i = 1:length(cor)
            fm = find(MC(:,i));
            for j1 = 1:length(fm)
                for j2 = j1+1:length(fm)
                    cor(i) = cor(i) + C_corr(fm(j1),fm(j2));
                end
            end
        end
        [~,ind] = sort(cor,'descend');
        nm = min(length(ind),cfg.mxcluster);   % number of merging operations
        merged_ROIs = cell(nm,1);
        for i = 1:nm
            merged_ROIs{i} = find(MC(:,ind(i)));
        end
        %
        ROI = [];
        singletons = setdiff(1:size(A,2),cell2mat(merged_ROIs));
        mergedA = false(size(A,1), length(merged_ROIs));
        for i =1:length(merged_ROIs)
            mergedA(:,i) = any(A(:,merged_ROIs{i}),2);
            ROI(i,:) = mean(C(merged_ROIs{i},:));
        end
        mergedA = [mergedA , logical(A(:,singletons))];
        ROI = [ROI; C(singletons,:)];
        % size thresholding
        sizeThresh = cfg.SizeThresh(1);
        maxSizeThresh = cfg.SizeThresh(2);
        inds = sum(mergedA)>sizeThresh & sum(mergedA)<maxSizeThresh ;
        mergedA = mergedA(:,inds);
        ROI = ROI(inds,:);
        
        n_clust = size(mergedA,2);
        %figure;imagesc(ROI);
        %  saveas(gcf, fullfile(cfg.outputfilepath,['ROI_temporal_',cfg.fig_str,'_nclust_',num2str(cfg.n_clust(iclust)),'_thrcluster_',num2str(cfg.thrcluster(ithrcluster)),'.png']));
        
        [~,~,~,~,sortInds] = order_ROIs(mergedA, ROI);
        mergedA = mergedA(:,sortInds);
        ROI = ROI(sortInds,:);
        
        %
        mergedA = full(mergedA>0);
        L=zeros(NROWS,NCOLS);
        for i=size(mergedA,2):-1:1
            L((mergedA(:,i))) = i;
        end
        %
        options.thr_method='max';
        options.plot_bck_image = true;
        if exist('stdA_all','var')
            %     figure;
            %    ha = tight_subplot(1,2,[.01 .05],[.1 .05],[.05 .02]);
            %   axes(ha(1));
            %  CC = plot_contours(mergedA,(mean(stdA_all,3)),options,1,100);
            % axes(ha(2));
            %imagesc(label2rgb(L));axis image
        else
            temp3d=zeros(NROWS,NCOLS,3);
            temp3d(:,:,2) = max_Psi;
            %figure;
            % ha = tight_subplot(1,2,[.01 .05],[.1 .05],[.05 .02]);
            %  axes(ha(1));
            %   CC = plot_contours(mergedA,temp3d*1/quantile(max_Psi(:),0.99),options,1,n_clust);
            
            %    axes(ha(2));
            %     imagesc(label2rgb(L));axis image
            %  end
            %   set(gcf,'Position',[38 374 962 420]);
            % saveas(gcf, fullfile(cfg.outputfilepath,['contours_',cfg.fig_str,'_nclust_',num2str(cfg.n_clust(iclust)),'_thrcluster_',num2str(cfg.thrcluster(ithrcluster)),'.png']));
            
            % plot the cell shape of all ROIs
            %  figure
            % for i = 1:min(size(mergedA,2),42)
            %    temp2=zeros(NROWS,NCOLS);
            %   temp2(mergedA(:,i)) = 1;
            %  props = regionprops(temp2,'BoundingBox');
            % subplot(6,7,i)
            %imagesc(temp2.*max_Psi);axis image
            %ylim([props.BoundingBox(2), props.BoundingBox(2)+props.BoundingBox(4)])
            %xlim([props.BoundingBox(1), props.BoundingBox(1)+props.BoundingBox(3)])
            %axis off
            %title(sprintf('Neuron %d',i));
            %end
            %set(gcf,'Position',[ 27    71   973   723]);
            %saveas(gcf, fullfile(cfg.outputfilepath,['ROIS_1_',cfg.fig_str,'_nclust_',num2str(cfg.n_clust(iclust)),'_thrcluster_',num2str(cfg.thrcluster(ithrcluster)),'.png']));
            
            %    if size(mergedA,2) > 42
            %        figure
            %        for j = 1:min(size(mergedA,2)-42,42)
            %            i = j + 42;
            %            temp2=zeros(NROWS,NCOLS);
            %            temp2(mergedA(:,i)) = 1;
            %            props = regionprops(temp2,'BoundingBox');
            
            %           subplot(6,7,j)
            %           imagesc(temp2.*max_Psi);axis image
            %           ylim([props.BoundingBox(2), props.BoundingBox(2)+props.BoundingBox(4)])
            %           xlim([props.BoundingBox(1), props.BoundingBox(1)+props.BoundingBox(3)])
            %           axis off
            %           title(sprintf('Neuron %d',i));
            %       end
            %   end
            %  set(gcf,'Position',[ 27    71   973   723]);
            %  saveas(gcf, fullfile(cfg.outputfilepath,['ROIS_2_',cfg.fig_str,'_nclust_',num2str(cfg.n_clust(iclust)),'_thrcluster_',num2str(cfg.thrcluster(ithrcluster)),'.png']));
            %% make sure there are no overlaps
            if ~cfg.isoverlap
                mergedA_resolved = mergedA;
                overlapheatmap = sum(mergedA, 2);
                pixels2resolve = find(overlapheatmap > 1);
                for cluster_i = 1:size(mergedA,2)
                    cluster_pixels = mergedA(:, cluster_i)>0;
                    cent(:, cluster_i) = mean(embedding_all(cluster_pixels,:),1);
                end
                for pix_i = 1:length(pixels2resolve)
                    currpix = pixels2resolve(pix_i);
                    clusters2resolve = find(mergedA(currpix, :));
                    D=pdist2(cent',embedding_all(currpix, :));
                    D(~mergedA(currpix, :)) = inf;
                    [~, mini] = min(D);
                    mergedA_resolved(currpix, :) = false(1, size(mergedA_resolved,2));
                    mergedA_resolved(currpix, mini) = true;
                end
                mergedA = mergedA_resolved;
                
                
            end
            
            % Check clustering @renu
            L1 = zeros(NROWS,NCOLS);
            for i= 1:size(mergedA,2)
                L1(mergedA(:,i)>0) = i;
            end
            figure;
            imagesc((L1==0));

            %% make sure each pixel is related to a specific parcel
            if cfg.nomissingpixels
                
                % Calculate a heatmap of missing pixels for each cluster
                missingheatmap = sum(mergedA, 2);
                
                % Identify pixels that are missing in the heatmap and belong to allregions
                missingheatmap = (missingheatmap == 0 & allregions(:) == 1);
                
                % Create a copy of the mergedA matrix for resolution
                mergedA_resolved = mergedA;
                
                % Find pixels that need to be resolved (@renu - modification)
                dists = (bwdist(L > 0) .* (allregions == 1));
                inds = find(dists(:) > 0);
                figure; imagesc(dists);
                zero_pixel_dists = dists(inds);
                [sorted_dists, sortinds] = sort(dists(inds));
                pixels2resolve = inds(sortinds);
                %pixels2resolve = find(missingheatmap == true); % old code
                
                % Loop through each cluster
                for cluster_i = 1:size(mergedA, 2)
                    % Find pixels belonging to the current cluster
                    cluster_pixels = mergedA(:, cluster_i) > 0;
                    
                    % Calculate the centroid of the embedding for cluster pixels
                    cent(:, cluster_i) = mean(embedding_all(cluster_pixels, :), 1);
                end
                
                % figure1 = figure;
                % figure2 = figure;
                % Loop through pixels that need to be resolved
                prev_num_pix2resolve = 0;
                while (~isempty(pixels2resolve)) % @renu
                    for pix_i = 1:length(pixels2resolve)
                        currpix = pixels2resolve(pix_i);
                        
                        % Convert current pixel index to subscripts (row, column)
                        [f, g] = ind2sub([NROWS, NCOLS], currpix);
                        
                        % Create a grid of neighboring pixel subscripts
                        [ff, gg] = meshgrid(f + [-1:1], g + [-1:1]);

                        % Ensure indices are within bounds
                        if any(ff(:) < 1 | ff(:) > NROWS) || any(gg(:) < 1 | gg(:) > NCOLS)
                            error('Indices in ff or gg are out of range.');
                        end
                        
                        % Convert neighboring subscripts to linear indices
                        nninds = sub2ind([NROWS, NCOLS], ff(:), gg(:));
                        
                        % Remove indices that correspond to non-region pixels
                        nninds = setdiff(nninds, find(allregions == false));
                        
                        % @renu visualize
                        % [f1, g1] = ind2sub([NROWS, NCOLS], nninds);
                        % vis_grid = zeros(NROWS, NCOLS);
                        % vis_grid(sub2ind([NROWS, NCOLS], find(allregions==true))) = 0.5;
                        % vis_grid(f1, g1) = 0.7;
                        % vis_grid(f, g) = 1;
                        % figure(figure1);
                        % imagesc(vis_grid, [0, 1]);
                        % colormap(figure1, 'parula');
                        
    
                        % [@renu] Calculate a heatmap of missing pixels for each cluster
                        missingheatmap = sum(mergedA_resolved, 2);
    
                        % [@renu] Identify pixels that are missing in the heatmap and belong to all regions
                        missingheatmap = (missingheatmap == 0 & allregions(:) == 1);
                        
                        % Check if all neighboring pixels are missing or have less than 4 missing pixels
                        %  @renu
                        LS_PIX = 4;
                        if (prev_num_pix2resolve == length(pixels2resolve))
                            LS_PIX = 1;
                        end
                        if all(missingheatmap(nninds)) || sum(missingheatmap(nninds) == 0) < LS_PIX  % @renu
                            continue;
                        end
                        
                        % Calculate pairwise distances between the centroid of clusters and the current pixel
                        D = pdist2(cent', embedding_all(currpix, :));
                        
                        % Find indices of clusters that contain the current pixel
                        [a, b] = find(mergedA_resolved(nninds, :)); % [@renu]
                        
                        % Identify unique clusters that contain the pixel
                        cand = unique(b);
                        bycand = zeros(1, size(mergedA_resolved, 2)); % [@renu]
                        bycand(cand) = true;
                        
                        % Update distances to be infinity for clusters not in 'cand'
                        D(~bycand) = inf;
                        
                        % Find the index of the cluster with the minimum distance
                        [~, mini] = min(D);
                        
                        % Mark the resolved pixel in the mergedA_resolved matrix
                        mergedA_resolved(currpix, mini) = true;
                    end

                    % @renu
                    L_t = zeros(NROWS,NCOLS);
                    for i0= 1:size(mergedA_resolved,2)
                        L_t(mergedA_resolved(:,i0)>0) = i0;
                    end
                    
                    % @renu
                    prev_num_pix2resolve = length(pixels2resolve);
                    dists = (bwdist(L_t > 0) .* (allregions == 1));
                    inds = find(dists(:) > 0);
                    [~, sortinds] = sort(dists(inds));
                    pixels2resolve = inds(sortinds);
                    % figure(figure2);
                    % imagesc(dists);
                    % all(L_t(sub2ind([NROWS,NCOLS], find(allregions==1))))
                    %pixels2resolve = find(missingheatmap == true);
                    
                end
                
                % Update the mergedA matrix with resolved pixels
                mergedA = mergedA_resolved;  
            end
            
            clear ROI_list;
            nROI=size(mergedA,2);
            if nROI>0
                for i=1:nROI
                    ROI_list(i)=struct('pixel_list',find(mergedA(:,i)==1),'name',['ROI',num2str(i)]);
                end
                % ROI_list=measure_ROI(A_full,ROI_list);
            else
                ROI_list=struct([]);
            end
            t4=toc;
            
            
            L = zeros(NROWS,NCOLS);
            for i= 1:size(mergedA,2)
                L(mergedA(:,i)>0) = i;
            end
            
            % % @renu test
            % dists1 = (bwdist(L > 0) .* (allregions == 1));
            % inds1 = find(dists1(:) > 0);
            % [sorted_dists1, sorted_inds1] = sort(dists(inds1));
            % figure; imagesc(dists1);

            % @renu - TemporalCorr
            clusterwise_within_corr = [];
            clusterwise_across_corr = [];
            if cfg.ComputeTemporalCorr
                % compute temporal correlation between pixels
                num_clusters = size(mergedA, 2);
                cluster_centres = zeros(num_clusters, 2);
                [nR, nC] = size(allregions);

                % within parcel 
                clusterwise_within_corr = zeros(1, num_clusters);
                %parfor cl = 1:num_clusters  % @renu check the PixxTime_dff for parfor 
                for cl = 1:num_clusters
                    cl_pix_ids = find(mergedA(:, cl)); % pixel ids as per original image
                    [~, cl_pix_pos_mask] = ismember(cl_pix_ids, allregionspix); % position of corresponding pixel time series in maksed data - PixxTime_dff
                    pix_tseries = PixxTime_dff(cl_pix_pos_mask, :);
                    corr_within_mat = corr(pix_tseries');            

                    % Compute average correlation value
                    upT = triu(corr_within_mat, 1);
                    nzupT = upT(upT ~= 0);
                    clusterwise_within_corr(cl) = mean(nzupT);

                    [p_r, p_c] = ind2sub([nR, nC], cl_pix_ids);
                    cluster_centres(cl, 1) = mean(p_r);
                    cluster_centres(cl, 2) = mean(p_c);
                end
                
                % across parcels
                NUM_NEAREST = 2;
                cl_distances = pdist2(cluster_centres, cluster_centres);
                clusterwise_across_corr = zeros(1, num_clusters);
                for cl = 1:num_clusters
                    dists = cl_distances(cl, :);
                    [~, sorted_ids] = sort(dists);
                    nearest_nbrs = sorted_ids(2:NUM_NEAREST+1); % 1st one is always itself (zero dist)
                    
                    % Time series' of pixels in the current parcel
                    cl_pix_ids = find(mergedA(:, cl)); % pixel ids as per original image
                    [~, cl_pix_pos_mask] = ismember(cl_pix_ids, allregionspix); % position of corresponding pixel time series in maksed data - PixxTime_dff
                    pix_tseries = PixxTime_dff(cl_pix_pos_mask, :);
                    
                    cumul_corr = 0;
                    cumul_pix = 0;
                    for nn = 1:NUM_NEAREST
                        cl_pix_ids_nn = find(mergedA(:, nearest_nbrs(nn))); % pixel ids as per original image
                        [~, cl_pix_pos_mask_nn] = ismember(cl_pix_ids_nn, allregionspix); % position of corresponding pixel time series in maksed data - PixxTime_dff
                        pix_tseries_nn = PixxTime_dff(cl_pix_pos_mask_nn, :);

                        cross_corr_ = corr(pix_tseries', pix_tseries_nn');
                        cumul_corr = cumul_corr + sum(cross_corr_(:));
                        cumul_pix = cumul_pix + size(cross_corr_, 1)*size(cross_corr_, 2);
                    end
                    clusterwise_across_corr(cl) = cumul_corr / cumul_pix;
                end
            end

            save(fullfile(cfg.outputfilepath,[cfg.fig_str,'_nclust_',num2str(cfg.n_clust(iclust)),...
                '_thrcluster_',num2str(cfg.thrcluster(ithrcluster)),'_out.mat']), 'ROI', 'ROI_list','mergedA',...
                'NROWS','NCOLS','cfg','max_Psi','embedding_norm','t1','t2','t3','t4', ...
                'clusterwise_within_corr', 'clusterwise_across_corr');

            L = rot90(L, -1); % @renu
            figure;imagesc(label2rgb(L));
            %imagesc((L==0)); % @renu
            saveas(gcf, fullfile(cfg.outputfilepath,['ROIS_L_',cfg.fig_str,'_nclust_',num2str(cfg.n_clust(iclust)),...
                '_thrcluster_',num2str(cfg.thrcluster(ithrcluster)),'.png']));
            
        end
        mergedA_all{ithrcluster} = mergedA; % renu
        L_all{ithrcluster} = L;
    end
end
