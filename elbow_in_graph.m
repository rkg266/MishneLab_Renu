function [thresh,thresh_ind] = elbow_in_graph(D1_sorted,doHalf)

if ~exist('doHalf','var')
    doHalf = true;
end

if doHalf
    n_points = round(length(D1_sorted)/2);
else
    n_points = length(D1_sorted);
end
points = [1:n_points; D1_sorted(1:n_points)']';

%# pull out first point
firstPoint = points(1,:);
lineVec = points(end,:) - firstPoint;

%# normalize the line vector
lineVecN = lineVec / sqrt(sum(lineVec.^2));

%# find the distance from each point to the line:
%# vector between all points and first point
vecFromFirst = bsxfun(@minus, points, firstPoint);
scalarProduct = vecFromFirst*lineVecN';
vecToLine = vecFromFirst - scalarProduct * lineVecN;

%# distance to line is the norm of vecToLine
distToLine = (sum(vecToLine.^2,2));
%# now all you need is to find the maximum
[~,thresh_ind] = max(distToLine);
thresh = D1_sorted(thresh_ind);

return
figure;
plot(1:n_points,D1_sorted(1:n_points))
hold on
plot(1:n_points,distToLine)
