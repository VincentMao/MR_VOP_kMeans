function [ B1_sens_lo, Ex_sens_lo, Ey_sens_lo, Ez_sens_lo, roi_mask, dim, Nc, FOV, d, md, mzd ] = mri_GetTipAngle_FEM( tipangle )
%MRI_GETTIPANGLE Summary of this function goes here
%   Detailed explanation goes here

%% Load sensitivities
% Sensitivities Courtesy of Steve Wright: 10th ISMRM, 2002, p. 854

%% FEM Simulation
% addpath('./FEM');
% % addpath('./Ref');
% % load fdtdsens;
% load B1_sens_lo;
% load Ez_sens_lo;

%% FDTD Simulation
addpath('./FDTD');
load('B1_sens_lo.mat')
load('Ex_sens_lo.mat')
load('Ey_sens_lo.mat')
load('Ez_sens_lo.mat')
max_Bxy = max(abs(B1_sens_lo(:)));
B1_sens_lo = 1/max_Bxy * B1_sens_lo;
max_Exy = max(max([abs(Ex_sens_lo(:)) abs(Ey_sens_lo(:))]));
Ex_sens_lo = 1/max_Exy * Ex_sens_lo;
Ey_sens_lo = 1/max_Exy * Ey_sens_lo;
Ez_sens_lo = 1/max_Exy * Ez_sens_lo;

FOV = 24; % 240 mm FOV

% mask = sum(abs(sens),3) > 0.1*max(col(sum(abs(sens),3)));
% B1_sens_lo = sens;
dim = size(B1_sens_lo,1); % dimension of square x-y grid
Nc = size(B1_sens_lo,3); % number of Tx coils

roi = image_geom('nx', dim, 'ny', dim, 'fov', FOV); % cm
% roi_mask = ellipse_im(roi, [0 0 2 3 0 1]) > 0;
% roi_mask = ellipse_im(roi, [0 0 10 10 0 1]) > 0;
% roi_mask = ellipse_im(roi, [0 0 11 8.5 0 1]) > 0; % for shepp-logan
% roi_mask = ellipse_im(roi, [0 0 12 12 0 1]) > 0; % for new image domain
% roi_mask = ellipse_im(roi, [0 0 8 7 0 1]) > 0; % for the loaded Duke phantom, in FEM simulation
roi_mask = ellipse_im(roi, [0 0 10 9.5 0 1]) > 0; % for the loaded Duke phantom, in FDTD simulation

%% Get the desired flip angle pattern

[xx,yy]=ndgrid(-FOV/2:FOV/dim:FOV/2-FOV/dim);
% d = phantom('Modified Shepp-Logan', dim);
% d = double(abs(xx).^2 + abs(yy).^2 <= 40); % the desired pattern
% d = double((abs(xx)<=10/2)&(abs(yy)<=5/2)); % the desired pattern
d = double(abs(xx).^2 / (10^2) + abs(yy).^2 / (9.5^2) <= 1); % for the loaded Duke phantom;

% blur the desired pattern to reduce gibbs ringing
d = conv2(d,exp(-(xx.^2+yy.^2)/0.25),'same');
d = d./max(abs(d(:)));
d = d .* roi_mask;
% d = 1 - d;

% magnetization-domain desired patterns
md = sin(d*tipangle*pi/180);
mzd = cos(d*tipangle*pi/180);

return;

end

