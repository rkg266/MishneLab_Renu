function dice_overall = multiclass_dice_coefficient(P1, P2) % renu
    % Get unique classes from both P1 and P2
    classes = unique([P1(:); P2(:)]);
    classes(classes == 0) = []; % Remove the background class if it's represented by 0

    % Initialize sums for intersections and total pixels
    intersection_sum = 0;
    total_p1_sum = 0;
    total_p2_sum = 0;

    % Loop over each class to calculate intersection and union
    for i = 1:length(classes)
        class = classes(i);
        p1_class = (P1 == class);
        p2_class = (P2 == class);

        intersection_sum = intersection_sum + sum(p1_class(:) & p2_class(:));
        total_p1_sum = total_p1_sum + sum(p1_class(:));
        total_p2_sum = total_p2_sum + sum(p2_class(:));
    end

    % Calculate the overall Dice coefficient
    dice_overall = (2 * intersection_sum) / (total_p1_sum + total_p2_sum);
end