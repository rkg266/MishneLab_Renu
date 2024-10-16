clc;
clear; 
close all;

%% To generate examples, borders around the clusters 

% Results diectory
resultDir = 'D:\UCSD_Acads\ProfGal_Research\test_run_norm1_pca0_kNN16_sftune4';

sesResultDir = fullfile(resultDir, 'run_fmri_sessions');
eodResultDir = fullfile(resultDir, 'run_fmri_evenodd\lssc_processed');

sesList = dir(fullfile(sesResultDir, '*.mat'));
eodList = dir(fullfile(eodResultDir, '*.mat'));

% process one file
FILEID = 1;
sesfileName1 = sesList(FILEID).name;
sesfileName2 = sesList(FILEID+1).name;
eodfileName = eodList(FILEID+2).name; 

sesResult1 = load(fullfile(sesResultDir, sesfileName1));
sesResult2 = load(fullfile(sesResultDir, sesfileName2));
eodResult = load(fullfile(eodResultDir, eodfileName));

%% Processing across-session files
labels_1 = sesResult1.labels{1};
labels_2 = sesResult2.labels{1};

% Cluster matching
pairing = pairComponents(sesResult1.mergedA{1}, sesResult2.mergedA{1});
p_1 = pairing.p1;
p_2 = pairing.p2;

% Re-labeling labels for 2nd one
for k1=1:size(labels_2, 1)
    for k2=1:size(labels_2, 2)
        if (labels_2(k1, k2) ~= 0)
            tp = find(p_2 == labels_2(k1, k2));
            if (isempty(tp))
                labels_2(k1, k2) = 0;
            else
                labels_2(k1, k2) = p_1(tp);
            end
            
        end
    end
end

dice_similarity1 = multiclass_dice_coefficient(labels_1, labels_2);

figure;
subplot(1, 2, 1);
imagesc(label2rgb(labels_1));
title('sub-SLC01\_session-1');
% Find unique clusters excluding zero (if zero is background)
uniqueClusters = unique(labels_1);
uniqueClusters(uniqueClusters == 0) = [];  % Remove zero if it's not a cluster

%% Overlay boundaries for each cluster
hold on;
for k = uniqueClusters'
    % Detect boundaries for the current cluster
    bw = labels_1 == k;
    boundaries = bwboundaries(bw);

    % Plot the boundaries
    for b = 1:length(boundaries)
        boundary = boundaries{b};
        plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 0.25); % 'k' for black color
    end
end
hold off;


subplot(1, 2, 2);
imagesc(label2rgb(labels_2));
title('sub-SLC01\_session-2');
% Find unique clusters excluding zero (if zero is background)
uniqueClusters = unique(labels_2);
uniqueClusters(uniqueClusters == 0) = [];  % Remove zero if it's not a cluster

% Overlay boundaries for each cluster
hold on;
for k = uniqueClusters'
    % Detect boundaries for the current cluster
    bw = labels_2 == k;
    boundaries = bwboundaries(bw);

    % Plot the boundaries
    for b = 1:length(boundaries)
        boundary = boundaries{b};
        plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 0.25); % 'k' for black color
    end
end
hold off;

%% Processing even-odd files
labels_1 = eodResult.even.labels{1};
labels_2 = eodResult.odd.labels{1};

pairing = pairComponents(eodResult.even.mergedA{1}, eodResult.odd.mergedA{1});
p_1 = pairing.p1;
p_2 = pairing.p2;

% Re-labeling labels for 2nd one
FLAG = max(p_1) + 1;
for k1=1:size(labels_2, 1)
    for k2=1:size(labels_2, 2)
        if (labels_2(k1, k2) ~= 0)
            tp = find(p_2 == labels_2(k1, k2));
            if (isempty(tp))
                labels_2(k1, k2) = 0;
                %labels_2(k1, k2) = FLAG;
                %FLAG = FLAG + 1;
            else
                labels_2(k1, k2) = p_1(tp);
            end
            
        end
    end
end

dice_similarity2 = multiclass_dice_coefficient(labels_1, labels_2);

figure;
subplot(1, 2, 1);
imagesc(label2rgb(labels_1));
title('sub-SLC01\_session-1\_run-11\_EVEN');
% Find unique clusters excluding zero (if zero is background)
uniqueClusters = unique(labels_1);
uniqueClusters(uniqueClusters == 0) = [];  % Remove zero if it's not a cluster

% Overlay boundaries for each cluster
hold on;
for k = uniqueClusters'
    % Detect boundaries for the current cluster
    bw = labels_1 == k;
    boundaries = bwboundaries(bw);

    % Determine the line style based on the cluster index
    if k <= max(p_1)
        lineStyle = '-';  % Solid line for positive cluster indices
    else
        lineStyle = '--'; % Dotted line for negative cluster indices
    end

    % Plot the boundaries
    for b = 1:length(boundaries)
        boundary = boundaries{b};
        plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 0.25); % 'k' for black color
    end
end
hold off;


subplot(1, 2, 2);
imagesc(label2rgb(labels_2));
title('sub-SLC01\_session-1\_run-11\_ODD');
% Find unique clusters excluding zero (if zero is background)
uniqueClusters = unique(labels_2);
uniqueClusters(uniqueClusters == 0) = [];  % Remove zero if it's not a cluster

% Overlay boundaries for each cluster
hold on;
for k = uniqueClusters'
    % Detect boundaries for the current cluster
    bw = labels_2 == k;
    boundaries = bwboundaries(bw);

    % Determine the line style based on the cluster index
    if k <= max(p_1)
        lineStyle = '-';  % Solid line for positive cluster indices
    else
        lineStyle = '--'; % Dotted line for negative cluster indices
    end

    % Plot the boundaries
    for b = 1:length(boundaries)
        boundary = boundaries{b};
        plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 0.25); % 'k' for black color
    end
end
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function dice_overall = multiclass_dice_coefficient(P1, P2) % renu
%     % Get unique classes from both P1 and P2
%     classes = unique([P1(:); P2(:)]);
%     classes(classes == 0) = []; % Remove the background class if it's represented by 0
% 
%     % Initialize sums for intersections and total pixels
%     intersection_sum = 0;
%     total_p1_sum = 0;
%     total_p2_sum = 0;
% 
%     % Loop over each class to calculate intersection and union
%     for i = 1:length(classes)
%         class = classes(i);
%         p1_class = (P1 == class);
%         p2_class = (P2 == class);
% 
%         intersection_sum = intersection_sum + sum(p1_class(:) & p2_class(:));
%         total_p1_sum = total_p1_sum + sum(p1_class(:));
%         total_p2_sum = total_p2_sum + sum(p2_class(:));
%     end
% 
%     % Calculate the overall Dice coefficient
%     dice_overall = (2 * intersection_sum) / (total_p1_sum + total_p2_sum);
% end