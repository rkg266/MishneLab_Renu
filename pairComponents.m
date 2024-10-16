function varargout = pairComponents(X1,X2,varargin)

% varargout = pairComponents(X1,X2,varargin)
% 
% Function to pair off components of two different dictionaries X1 and X2. 
% 
% 2019 - Adam Charles & Gal Mishne

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input testing

sz1 = size(X1);                                                            % Get the size of the data
sz2 = size(X2);                                                            % Get the size of the data

if nargin > 2
    pair_meth = varargin{1};
else
    pair_meth = 'max_corr';
end

% if ~strcmp(pair_meth ,'max_corr')
%     if numel(sz1)~=size(sz2)
%         error('sizes of two arrays do not match!')
%     end
%     if any(sz1~=sz2)
%         error('sizes of two arrays do not match!')
%     end
% end

X1 = reshape(X1,[],sz1(end));                                              % Reshape the data into a set of vectors
X2 = reshape(X2,[],sz2(end));                                              % Reshape the data into a set of vectors
X1 = X1*diag(1./sqrt(sum(X1.^2,1)));                                       % Normalize the data
X2 = X2*diag(1./sqrt(sum(X2.^2,1)));                                       % Normalize the data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create initial confusion matrix

switch pair_meth
    case 'max_corr'                                                        % If the max-correlation is selected:
        C = X1'*X2;                                                        % Calculate the correlation matrix     
        [p1, p2] = corrMatchingGreedy(C);                                  % In a greedy fashion, select pairs of matching coefficients
    case 'max_corrCoef'                                                    % If the max-correlation-coefficient is selected:
        C = diag(1./sqrt(sum(X1.^2,1)))*X1.'*X2*diag(1./sqrt(sum(X2.^2,1)));% Calculate the correlation coefficient matrix 
        [p1, p2] = corrMatchingGreedy(C);                                  % In a greedy fashion, select pairs of matching coefficients
    case 'hungarian'                                                       % If the hungarian algorithm method of pairing is selected: 
        [p2,~]   = munkres(-corr(X1,X2));                                  % 
        p1       = 1:numel(p2);                                            % 
        C = X1'*X2;                                                        % Save a backup of the correlation matrix
    case 'lin_fit'
        p1 = 1:size(X1,2);                                                 % Get the number of elements in the first dictionary
        p2 = 1:size(X2,2);                                                 % Get the number of elements in the second dictionary
        C = X2\X1;                                                         % Calculate the linear fit of each 
    case 'lsqr_match'                                                      % If the least-squares matching criterion is selected:
        C = X2\X1;                                                         % Calculate the correlation matrix     
        [p1, p2] = corrMatchingGreedy(C);                                  % In a greedy fashion, select pairs of matching coefficients
    otherwise
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output parsing

if nargout == 1
    out_struct.p1 = p1;
    out_struct.p2 = p2; % why in some cases, zeros are coming for 'hungarian'?
    out_struct.C = C(p1,p2);
    varargout{1} = out_struct;
elseif nargout == 2
    varargout{1} = p1;
    varargout{2} = p2;
elseif nargout == 3
    varargout{1} = p1;
    varargout{2} = p2;
    varargout{3} = C(p1,p2);
elseif nargout > 3
    varargout{1} = p1;
    varargout{2} = p2;
    varargout{3} = C(p1,p2);
    varargout{4} = C;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extra function

function [p1, p2] = corrMatchingGreedy(C)

p1       = [];                                                             % Initialize the row matches
p2       = [];                                                             % Initialize the column matches
for kk = 1:min(size(C,1),size(C,2))                                        % For every index...
    Cmax   = max(C(:));                                                    %   ... calculate maximum correlation...
    [I,K]  = find(C==Cmax,1,'first');                                      %   ... find the location of this maximum...
    p1     = cat(1,p1,I);                                                  %   ... add the row...
    p2     = cat(1,p2,K);                                                  %   ... and column to the paired indices...
    C(I,:) = -Inf;                                                         %   ... and finally remove all possible other pairings by setting the corresponding row...
    C(:,K) = -Inf;                                                         %   ... and column to -infinity so that it's never selected again.
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
