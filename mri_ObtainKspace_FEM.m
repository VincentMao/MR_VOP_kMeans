function [ ig, fmap, zmap, kspace, omega, wi_traj ] = mri_ObtainKspace_FEM( desired, ROI_mask, FOV, factor )
%MRI_OBTAINKSPACE Summary of this function goes here
%   Detailed explanation goes here

% Read the true image from the reference image
% xtrue magnitude data
[nx ny] = size(desired);
ig = image_geom('nx', nx, 'ny', ny, 'fov', FOV); % cm
ig.mask = ROI_mask;
pr sum(desired(ig.mask) ~= 0)

% Phase data
% Ignore the phase date in this scenario
fmap = [];
zmap = 0 + (2i*pi) * fmap;

% k-space trajectory
N = [nx ny];
f.traj = 'spiral_VM'; f.dens = {'voronoi'};
sampfactor = 1 ./ factor; % for every transmit coil, we should change its sampfactor to 1/factor
% Random Sampling
% samp = randsrc(ig.ny,1,[0 1; (1 - sampfactor) sampfactor]);
samp = rand(ig.ny,1) < sampfactor; % for EPI only
% All concentrate in the middle
% samp = [zeros(1, floor(ig.ny-sum(samp))) ones(1, sum(samp)) zeros(1, ceil(ig.ny-sum(samp)))]';
% samp = repmat([1 0]', ig.ny/factor, 1);
% samp = true(ig.ny,1); % fully sampled
printm('%% samples used: %g', sum(samp) / length(samp) * 100)
% [kspace omega wi_traj] = mri_trajectory(f.traj, {'samp', samp}, N,
% ig.fov, f.dens); % for EPI 
[kspace omega wi_traj] = mri_trajectory(f.traj, {'sampfactor', sampfactor}, N, ig.fov, f.dens); % for spiral

end
