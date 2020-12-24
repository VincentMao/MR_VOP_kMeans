function [ SAR_cluster, numCluster, matrix_Q_10g_VOP, core_idx ] = Clustering_VOP_10g_new( matrix_Q_10g, similiarity, slice, sizeX, sizeY, overestimate )

% VOP Clustering Algorithm 
% num = number of pre-defined clusters

Nc = size(matrix_Q_10g, 1);
% similiarity = zeros(sizeX*sizeY, numS);
SAR_cluster = zeros(sizeX, sizeY);

%% VOP clustering, Obtain the resulting number of clusters
% % sort the SAR matrices in terms of eigenvalues
% for obs = 1: numS
%     for k = 1: sizeX*sizeY
%         % Clustering based on the dominant eigenvalue
%         [dummy, similiarity(k,obs)] = eigs(abs(matrix_Q_10g(:,:,obs,k)), 1); % Find the dominant eigenvalue
%     end
% end
% only use the 79th slice for clustering
[dummy idxMaxtoMin] = sort(similiarity(:,slice), 'descend');
epsilon = overestimate;
numCluster = 1;
core_idx = idxMaxtoMin(1);
SAR_cluster(core_idx) = numCluster;
matrix_Z = zeros(Nc, Nc, sizeX*sizeY);
for k = 2: sizeX*sizeY
    printf('%d / %d', k, sizeX*sizeY);
    idx_k = idxMaxtoMin(k);
    currentQ = matrix_Q_10g(:,:,slice,idx_k);
    if (isequal(currentQ, zeros(Nc, Nc)))
        SAR_cluster(idx_k) = 0;
        continue;
    end
    % check to see if there exits ||Z|| <= epsilon that Q(k)+Z-core is psd
    % Need to check with all the previous VOPs
    for ii = 1: numCluster
        c_idx = core_idx(ii);
        core = matrix_Q_10g(:,:,slice,c_idx);
        [ Z, exitflag ] = FindPSD(currentQ, core, matrix_Z(:,:,ii), epsilon, Nc);
        if (exitflag == 1)
           if (norm(Z) >= norm(matrix_Z(:,:,ii))) % could replace with Z=matrix_Z(:,:,ii);
                matrix_Z(:,:,ii) = Z;
           end
           SAR_cluster(idx_k) = ii;
           break;
        end
    end
    if (SAR_cluster(idx_k) == 0)
        numCluster = numCluster + 1;
        SAR_cluster(idx_k) = numCluster;
        core_idx = [core_idx; idx_k];
    end
    printf('Current clusters: %d', numCluster);
end

%% Determine the resulting VOPs
matrix_Q_10g_VOP = zeros(Nc, Nc, numCluster);
for ii = 1: numCluster
    matrix_Q_10g_VOP(:,:, ii) = matrix_Q_10g(:, :, slice, core_idx(ii)) + matrix_Z(:,:,ii);
end

end

%% Find if there exists ||Z|| <= epsilon that Q+Z-B is psd
function [ Z, exitflag ] = FindPSD(B_current, B_core, currentZ, epsilon, Nc)
    % disp 'Finding the PSD matrix Z';
    Z = zeros(Nc, Nc);
    maxInters = 300;
    exitflag = 0;
    B_core = B_core + currentZ;
    for ii = 1: maxInters
        % printf('%d / %d', ii, maxInters);
        alleigs = eig(B_core+Z-B_current);
        alleigs = [alleigs; epsilon-norm(currentZ+Z)];
        if (epsilon-norm(currentZ+Z) < 0)
            break;
        end
        % if (all(alleigs >=0) || all(-real(alleigs(alleigs<0)) <= 1e-10) )
        if (all(alleigs >=0))
            Z = currentZ+Z;
            exitflag = 1;
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