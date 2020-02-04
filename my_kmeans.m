%% Implement the k-means algorithm to cluster the SAR matrices
function [SAR_cluster, CENTS] = my_kmeans(matrix_Q, similiarity_nan, core_idx, numCluster, maxiters)

% We only use one slice as one observation
% USAGE:
% [SAR_cluster, CENTS] = my_kmeans(squeeze(matrix_Q_10g(:,:,79,:)), similiarity_nan(:,79), core_idx, numCluster, maxiters)
Nc = size(matrix_Q, 1);
if Nc ~= size(matrix_Q, 2)
    error('matrix size mismatch!');
end
numPoints = size(matrix_Q, 3); % number of points

% CENTS = matrix_Q( :,:, ceil(rand(numCluster,1)*numPoints));                % Random location for Cluster Centers
CENTS = matrix_Q( :,:, core_idx);                                          % Initial location for Cluster Centers
CENTS_NEW = zeros(size(CENTS));                                            % Random location for Cluster Centers
DAL   = zeros(numPoints,numCluster+2);                                     % Distances and Labels

CurrentDist = Inf;
similiarity_nan(isnan(similiarity_nan)) = -Inf;
for n = 1:maxiters
   printf('%d / %d', n, maxiters);    
   %% Calculate the current label and Distance
   for i = 1:numPoints
      for j = 1:numCluster  
          DAL(i,j) = norm(FindPSD(matrix_Q(:,:,i), CENTS(:,:,j), Nc));     
      end
      [Distance, CN] = nanmin(DAL(i,1:numCluster));             % 1:K are Distance from Cluster Centers 1:K 
      if ~(isequal(matrix_Q(:,:,i), zeros(Nc, Nc)))
          DAL(i,numCluster+1) = CN;                             % K+1 is Cluster Label
      end
      DAL(i,numCluster+2) = Distance;                           % K+2 is Minimum Distance
   end
   
   %% Determine if use the last centroid or the new centroid
   printf('Current Dist: %f', CurrentDist);
   printf('NEW Dist: %f', max(DAL(:, numCluster+2)));
   CurrentLabel = DAL(:, numCluster+1);
   CurrentDist = nanmax(DAL(:, numCluster+2));
   
   %% Look for the next possible cluster centers
   for i = 1:numCluster
      A = find(CurrentLabel == i);                                                       % Cluster K Points
      if isempty(A)
         CENTS_NEW(:,:, i) = zeros(Nc, Nc);
      else 
         [dummy, idxMaxtoMin] = sort(similiarity_nan(A), 'descend');
         minlimit = min([100, size(idxMaxtoMin,1)]);
         CENTS_NEW(:,:,i) = FindMinDist(matrix_Q(:,:,A), idxMaxtoMin(1:minlimit), Nc);   % New Cluster Centers
      end
      while (isequal(CENTS_NEW(:,:,i), zeros(Nc, Nc)))                                   % If CENTS(:,:,i) Is zero Then Replace It With Random Point
         CENTS_NEW(:,:,i) = matrix_Q(:,:,randi(numPoints));
      end
   end
   if isequal(CENTS, CENTS_NEW)                                 % check to see if it converge
       break;
   else
       CENTS = CENTS_NEW;
   end
   
end

SAR_cluster = CurrentLabel;

end

%% Find if there exists Q+Z-B is psd
function [ Z ] = FindPSD(B_current, B_core, Nc)
    % disp 'Finding the PSD matrix Z'; 
    Z = zeros(Nc, Nc);
    maxInters = 300;
    for ii = 1: maxInters
        % printf('%d / %d', ii, maxInters);
        alleigs = eig(B_core+Z-B_current);
        if (all(alleigs >=0) || all(-real(alleigs(alleigs<0)) <= 1e-5) )
            break;
        else
            % Find the next possible Z
            [V, D] = eig(B_core+Z-B_current);
            D_orig = D;
            D(D<=0)=0;
            D_new = D - D_orig;
            B_new = V*D_new*V';
            Z = Z + B_new;
        end 
    end
end

%% Find minimum sum_r||Q(r)+Z-CENTS||
function [CENTS] = FindMinDist(matrix_Q, setA, Nc)
    % Find CENTS in the cluster such that sum_r||Q(r)+Z-CENTS|| is
    % minimum
    DAL = zeros(1, size(setA,1));
    currentMin = Inf;
    idx_min = 0;
    for ii = 1: size(setA,1)
        for jj = 1: size(matrix_Q,3)
            DAL(jj) = norm(FindPSD(matrix_Q(:,:,jj), matrix_Q(:,:,setA(ii)), Nc));
        end
        if currentMin > max(DAL(:))
            currentMin = max(DAL(:));
            idx_min = setA(ii);
        end
    end
    CENTS = matrix_Q(:,:,idx_min);
end