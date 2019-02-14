% This script can be used to test the integration of a tracker to the
% framework.

addpath('/home/lee/tracking/challenge/vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

tracker = tracker_load('SiamVGG');

workspace_test(tracker, sequences);

