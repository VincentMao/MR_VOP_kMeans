% This script is a simulation of MIMO MRI model with SAR contraints, it simulates the RF pulses
% for the the optimization of SAR and excitation accuracy of the model.

% Written by Vincent Xianglun Mao 10/2019
% History: 

%% Clean
clear all; clc; close all;
%% Load necessary toolbox

addpath('./toolbox'); % Jeffery Fessler's IRT toolbox
addpath(genpath('./minTimeGradient')); % mlustig toolbox
setup;

%% Load sensitivities and the desired flip angle pattern
% Get the desired flip angle pattern
tipangle = 90;
[B1_sens, Ex_sens, Ey_sens, Ez_sens, ROI_mask, dim, Nc, FOV, d, md, mzd ] = mri_GetTipAngle_FEM(tipangle);

% Load density and conductivity
load('Sigma_lo.mat')
load('Density_lo.mat')

%% Get the initial EPI k-space trajectory and its gradients
factor = 2; % acceleration factor
[ ig, fmap, zmap, kspace, omega, wi_traj ] = mri_ObtainKspace_FEM(d, ROI_mask, FOV, factor);

%% Setup the MIMO MRI Model, calculate the real k-space trajectory

% Calculate the gradient waveforms from the kspace trajectory
% GE MR750 50 mT/m 200T/m/s for 3T
% GE MR950 50 mT/m 200T/m/s for 7T
% GE Signa 7T 100 mT/m 200T/m/s for 7T
[Curve_rv, time_rv, grad_rv, s_rv, k_rv] = minTimeGradient([kspace 0*kspace(:,1)],0, 0, 0, 10, 20, 4e-3); % sampling rate 
% We only take kxy and Gxy
grad = grad_rv(:,1:2);
kspace = k_rv(:,1:2);
% Tikhonov regularization, 16 dof for spiral
NN = [size(kspace, 1)-16, 16];

%% Small tip angle approximation
% We will start with small-tip-angle approximation, will move on to the
% large-tip-angle approximation 

% The new goal for the small tip angle approximation: optimize the SNR in
% the receive side

% Obtain the new sensitivity map with mask
sens_mask = zeros(size(B1_sens));
for ii = 1:size(B1_sens,3)
    sens_mask(:,:,ii) = B1_sens(:,:,ii).*ig.mask;
end

% field-map params
Nt = size(kspace,1); % number of samples in pulses
dt = 4e-6; % seconds, RF and gradient sampling period
tt = 0:dt:(Nt-1)*dt;tt = tt-(Nt-1)*dt;L = 4;
% nufft params
J = 6;K = 2*dim;
nufft_args = {[dim dim],[J J],[K K],[dim dim]/2,'minmax:kb'};
gambar = 4257;             % gamma/2pi in Hz/g
gam = gambar*2*pi;         % gamma in radians/g
% trick: the system matrix is just the transpose of a SENSE image recon matrix!
Gsml = Gmri_SENSE(kspace, logical(ones(dim)),'fov',[FOV FOV],'basis',{'dirac'}, ...
                  'nufft',nufft_args,'exact',0, ...
                  'sens',conj(reshape(sens_mask,[dim*dim Nc]))*(-1i*gam*dt), ...
                  'ti',-tt,'L',L,'zmap',zmap)';
              
%% Obtain the VOP and k-means clustering SAR compression model

sizeX = dim;
sizeY = dim;
slice = 79; % choose the slice you wish to proceed
load('./Data/similiarity_nan.mat');
load('./Data/similiarity.mat');
overestimate = 0.05 * max(similiarity_nan(:, slice));
maxiters = 200;
[ SAR_cluster_VOP, numCluster, matrix_VOP, core_idx ] = Clustering_VOP_10g( matrix_Q_10g, similiarity, slice, sizeX, sizeY, overestimate );
[ SAR_cluster_kmeans, CENTS ] = my_kmeans(squeeze(matrix_Q_10g(:,:,slice,:)), similiarity_nan(:,slice), core_idx, numCluster, maxiters);
SAR_cluster_kmeans = reshape(SAR_cluster_kmeans, dim, dim);

matR_core = zeros(Nc, Nc, numCluster);
matrix_Z = zeros(Nc, Nc, numCluster);
Overestimation = zeros(1, numCluster);
OverestimationData = nan(sizeX*sizeY, numCluster);
for k = 1: numCluster
    % printf('%d / %d', k, numCluster);
    idx_k = find(SAR_cluster_kmeans==k);
    matR_core(:,:,k) = CENTS(:,:,k);
    for ii = 1: size(idx_k, 1)
        % printf('%d / %d', ii, size(idx_k, 1));
        currentQ = matrix_Q_10g(:,:,slice, idx_k(ii));
        Z_tmp = FindPSD(currentQ, matR_core(:,:,k), Nc);
        if norm(Z_tmp) > Overestimation(k)
            matrix_Z(:,:,k) = Z_tmp;
            Overestimation(k) = norm(Z_tmp);
        end
        OverestimationData(ii, k) = norm(Z_tmp) ./ max(similiarity_nan(:, slice)); 
    end
end
Overestimation = Overestimation ./ max(similiarity_nan(:, slice));

% Determine the resulting k-means clusters
matrix_kmeans = zeros(Nc, Nc, numCluster);
for ii = 1: numCluster
    matrix_kmeans(:,:, ii) = matR_core(:,:,ii) + matrix_Z(:,:,ii);
end
% optional, save all the data into files
% save('./Data/matrix_kmeans_79.mat', 'matrix_kmeans', '-v7.3');
% save('./Data/matrix_VOP_79.mat', 'matrix_VOP', '-v7.3');

%% Optimization with SAR and RF power constraints

% Global alpha
alpha = 1e-2;
niters = floor(Nt/4);

% % VOP model
% load('./Data/matrix_VOP_79.mat');
% mCluster = matrix_VOP;
% k-means model
load('./Data/matrix_kmeans_79.mat');
mCluster = matrix_kmeans;
numCluster = size(mCluster, 3);
% Global SAR constraints for simpilicity, can change the alpha values for
% each cluster
alpha_local = 1e-1 * ones(1, numCluster);
mCluster_sum = zeros(Nc, Nc);
for ii = 1: numCluster
    mCluster_sum = mCluster_sum + alpha_local(ii) * mCluster(:,:, ii);
end
selectvec = zeros(1, Nc*Nt);
for k = 1: Nt
   selectvec(Nc*(k-1)+1:Nc*k) = [0:(Nc-1)] .* Nt + k.*ones(1, Nc); 
end
selectmat = sparse(Nc*Nt, Nc*Nt);
for k = 1: Nc*Nt
    selectmat(k, selectvec(k)) = 1;
end
disp('Cluster Processing');
% Without local SAR constraint and log barrior
Ar = repmat(mCluster_sum, 1, Nt);
Ac = mat2cell(Ar, size(mCluster_sum,1), repmat(size(mCluster_sum,2),1, Nt));
tmp = blkdiag(Ac{:});  
matQ = selectmat' * tmp * selectmat;
matQall = (1/Nt)*matQ; % add the tuning parameter

% Compute the ldl decomposition of matQ
matQall(logical(eye(Nc*Nt))) = real(diag(matQall));
[L, D] = ldl(matQall);
matC = (L*sqrt(D))';
clear L D Ar Ac tmp mCluser_sum;

% Design the RF pulse, with power control
% get the tikhonov penalty vector
beta = 5e-4;
betavec = ones(Nc*Nt,1)*sqrt(beta);
% penalize the rewinder to force it to zero
betavec = betavec+1000000*kron(ones(Nc,1),[zeros(NN(1),1);ones(NN(2),1)]);
matC = matC + diag(betavec);

disp 'Designing the ultimate RF pulse, with SAR constraint'
[xS,info] = qpwls_pcg(zeros(Nc*Nt,1),Gsml,1,d*tipangle*pi/180,0, ...
		      matC,1,niters,ones(size(d)));
B1_optimal(:,1) = xS(:,end);

% Save the excitation pattern and the SAR map
Excitation = sin(reshape(abs(Gsml*B1_optimal).*ROI_mask(:), dim, dim));
NRMSE = nrmse(abs(d*tipangle*pi/180).*ROI_mask, abs(Gsml*B1_optimal).*ROI_mask(:));


%% Excitation Results

% Plot the 1d profile of the Excitation Pattern
plot(linspace(-FOV/2, FOV/2, 128), abs(md(:, 64)), 'r-', 'MarkerSize', 15, 'Linewidth', 1.6);
hold on;
plot(linspace(-FOV/2, FOV/2, 128), abs(Excitation(:,64)), 'b--','MarkerSize', 15, 'Linewidth', 1.6);
hold off; axis square;
axis([-FOV/2 FOV/2 0 1.1]); set(gca, 'FontSize', 14);
legend({'Desired Pattern', 'pTx + SAR Compress'}, 'Interpreter','latex');

% mesh plot of the excitation
[xx,yy]=ndgrid(-FOV/2:FOV/dim:FOV/2-FOV/dim);
mesh(xx,yy,abs(Excitation));
%title('Simulated Excitation Pattern (Mesh Plot)', 'Interpreter','latex');
axis([-FOV/2 FOV/2 -FOV/2 FOV/2 -1 1]);
xlabel('x (cm)', 'Interpreter','latex', 'FontSize', 18);
ylabel('y (cm)', 'Interpreter','latex', 'FontSize', 18);
zlabel('$M_{xy}$', 'Interpreter','latex', 'FontSize', 18);
zlim([0, 1]);

%% Plot the N-channel RF Pulses

figure
% plot channel 1's pulses
subplot(121)
plot(0:dt*1000:(NN(1)-1)*dt*1000,abs(B1_optimal(1:NN(1),1)),'r');
xlabel 'Time (ms)'
ylabel '|b_1(t)| (a.u.)'
axis([0 (NN(1)-1)*dt*1000 0 max(abs(B1_optimal(:,1)))]);
subplot(122)
plot(0:dt*1000:(NN(1)-1)*dt*1000,angle(B1_optimal(1:NN(1),1)),'r');
xlabel 'Time (ms)'
ylabel '\angle b_1(t) (Radians)'
axis([0 (NN(1)-1)*dt*1000 -pi pi]);
legend('pTx + SAR Compress');


