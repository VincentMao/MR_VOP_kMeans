function [ SAR_cluster, numCluster, matrix_Q_brain_VOP, core_idx ] = Clustering_VOP_brain( matrix_Q_10g, similiarity, sizeX, sizeY, numS, overestimate )

% VOP Clustering Algorithm 
% num = number of pre-defined clusters

Nc = size(matrix_Q_10g, 1);
SAR_cluster = zeros(sizeX*sizeY, numS);

%% VOP clustering, Obtain the resulting number of clusters
[dummy idxMaxtoMin] = sort(similiarity(:), 'descend');
[row, column] = ndgrid(1:size(similiarity, 1),1:size(similiarity,2));
epsilon = overestimate;
numCluster = 1;
% core_idx numCluster * [idx, slice]
core_idx = [row(idxMaxtoMin(1)), column(idxMaxtoMin(1))];
SAR_cluster(core_idx(1), core_idx(2)) = numCluster;
matrix_Z = zeros(Nc, Nc, sizeX*sizeY); % size big enough
for k = 2: sizeX*sizeY*numS
    printf('%d / %d', k, sizeX*sizeY*numS);
    idx_k = [row(idxMaxtoMin(k)), column(idxMaxtoMin(k))];
    currentQ = matrix_Q_10g(:,:,idx_k(2),idx_k(1));
    if (isequal(currentQ, zeros(Nc, Nc)))
        SAR_cluster(idx_k(1), idx_k(2)) = 0;
        continue;
    end
    % check to see if there exits ||Z|| <= epsilon that Q(k)+Z-core is psd
    % Need to check with all the previous VOPs
    for ii = 1: numCluster
        c_idx = core_idx(ii, :);
        core = matrix_Q_10g(:,:,c_idx(2),c_idx(1));
        [ Z, exitflag ] = FindPSD(currentQ, core, matrix_Z(:,:,ii), epsilon, Nc);
        if (exitflag == 1)
            if (norm(Z) >= norm(matrix_Z(:,:,ii)))
                matrix_Z(:,:,ii) = Z;
            end
            SAR_cluster(idx_k(1), idx_k(2)) = ii;
            break;
        end
    end
    if (SAR_cluster(idx_k(1), idx_k(2)) == 0)
        numCluster = numCluster + 1;
        SAR_cluster(idx_k(1), idx_k(2)) = numCluster;
        core_idx = [core_idx; idx_k];
    end
    printf('Current clusters: %d', numCluster);
end

%% reshape the segmentation results
SAR_cluster = reshape(SAR_cluster, sizeX, sizeY, numS);

%% Determine the resulting VOPs
matrix_Q_brain_VOP = zeros(Nc, Nc, numCluster);
for ii = 1: numCluster
    matrix_Q_brain_VOP(:,:, ii) = matrix_Q_10g(:, :, core_idx(ii,2), core_idx(ii,1)) + matrix_Z(:,:,ii);
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