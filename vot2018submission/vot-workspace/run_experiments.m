% This script can be used to execute the experiments for a single tracker
% You can copy and modify it to create another experiment launcher

addpath('/home/lee/tracking/challenge/vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();
%experiments{1,1}.parameters.repetitions = 1;

tracker = tracker_load('SiamVGG');

workspace_evaluate(tracker, sequences, experiments);

