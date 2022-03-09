% Script to denoise data from event camera using CNN
% R. Wes Baldwin
% University of Dayton
% June 2020

% Please cite the CVPR 2020 paper: 
%  "Event Probability Mask (EPM) and Event Denoising Convolutional 
%   Neural Network (EDnCNN) for Neuromorphic Cameras"

% This code uses functions from Garrick Orchard's "Motion-Ground-Truth"
% Available here: https://github.com/gorchard/Motion-Ground-Truth

% DVSNOISE20 data can be downloaded from:
% (https://sites.google.com/a/udayton.edu/issl/software/dataset)

% This script assumes you have the data in MAT format (2_mat.zip) from
% DVSNOISE20 website. If you need to convert your data use one of these two
% methods...
% 1. Convert AEDAT data into a MAT file use AEDAT-TOOLS
% (https://gitlab.com/inivation/AedatTools)
% 2. Convert AEDAT4 data into a MAT file use aedat4tomat
% (https://github.com/bald6354/aedat4tomat)

% The folder "camera" contains the camera parameters to remove lens distortion
% and fixed pattern noise for the DAVIS 346 camera we used for the
% DVSNOISE20 dataset. You can generate your own distortion correction for a
% different camera using the built-in MATLAB calibration application.

% This code is not optimized for speed (i.e. loading and saving data to
% disk), but rather designed to isolate functions and make the code easier
% to follow or partially reuse. Please contact me at baldwinr2@udayton.edu
% with any questions. Thanks!

%% Setup and Variables
clear, clc, close all
set(0,'DefaultFigureWindowStyle','docked')

% Path to data files in MAT format (put files you wish to process here)
mainDir = '2_mat\'

% Path to output directory (results go here)
outDir = 'edncnn_output\'

% Settings
inputVar.depth = 2; %feature depth per polarity (k in paper)
inputVar.neighborhood = 12; %feature neighborhood (m in paper)(0=1x1, 1=3x3, 2=5x5, etc.)
inputVar.maxNumSamples = 10e3; %Only sample up to this many events per file for pos and neg polarities combined
inputVar.waitBuffer = 2; %time in seconds to wait before sampling an event - early events have no history & dvs tends to drop the feed briefly in the first second or so
inputVar.minTime = 150; %any amount less than 150 microseconds can be ignored (helps with log scaling) (feature normalization)
inputVar.maxTime = 5e6; %any amount greater than 5 seconds can be ignored (put data on fixed output size) (feature normalization)
inputVar.maxProb = 1; %"probability" score capped at this number
inputVar.nonCausal = false; %if true, double feature size by creating surface both back in time AND forward in time (not used in paper)
inputVar.removeGyroBias = 0.5; %use the first 0.5 seconds of data to zero the gyro
inputVar.writeOutGIF = false; %write out an animated gif of the EPM labels assigned to the events

% Camera/Lens details - DAVIS346
inputVar.focalLength = 4; %mm (4.5=240C, 4=346(wide), 12=346(zoom)
inputVar.Wx = 6.4;  %width of focal plane in mm
inputVar.Wy = 4.8;  %height of focal plane in mm
load('camera\fpn_346.mat')
inputVar.fpn = fpn; %fixed pattern noise
clear fpn

% Add EDnCNN code to path
addpath('code')

    
%% Process each file and calculate EPM

% Gather a list of files and 
files = dir([mainDir '*.mat']);


%% Use network to predict data labels (real/noise)

% Gather a list of files 
files = dir([outDir '*epm.mat']);

for fLoop = 1:numel(files)
    
    %DVSNOISE20 has 3 datasets per scene (group)
    grpLabel = floor((fLoop-1)/3) + 1;

    file = [outDir files(fLoop).name]
    [fp,fn,fe] = fileparts(file);
    
    load(file, 'aedat', 'inputVar')
    % load([outDir num2str(grpLabel) '_trained_v1.mat'], 'net')
    
    YPred = makeLabeledAnimationsSave(aedat, inputVar);

    save([outDir fn '_pred.mat'],'YPred','-v7.3')
    
end

% %% Score results using RPMD

% files = dir([outDir '*epm.mat']);

% for fLoop = 1:numel(files)

%     file = [outDir files(fLoop).name]
%     [fp,fn,fe] = fileparts(file);
    
%     load(file, 'aedat')
%     load([outDir fn '_pred.mat'],'YPred')
    
%     YPred = YPred(:,1); %please make sure you need column #1 and not column #2 (column #2 is 1-column #1)(This may change based on how matlab auto assigns output in the network classification layer.)
    
%     [noisyScore(fLoop), denoiseScore(fLoop)] = scoreDenoise(aedat, YPred);
    
% end

% %Average results for each scene and plot
% figure% bar(cat(1,mean(reshape(noisyScore,3,[]),1),mean(reshape(denoiseScore,3,[]),1))')
% legend('Noisy','Denoised')
% xlabel('Scene')
% ylabel('RPMD')
