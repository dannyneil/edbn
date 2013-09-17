%% Load paths
addpath(genpath('.'));

%% Load data
load mnist_uint8;

% Convert data and rescale between 0 and 0.2
train_x = double(train_x) / 255 * 0.2;
test_x  = double(test_x)  / 255 * 0.2;
train_y = double(train_y) * 0.2;
test_y  = double(test_y)  * 0.2;

%% Train network
% Setup
rand('seed', 42);
clear edbn opts;
edbn.sizes = [784 100 10];
opts.numepochs = 6;

[edbn, opts] = edbnsetup(edbn, opts);

% Train
fprintf('Beginning training.\n');
edbn = edbntrain(edbn, train_x, opts);
edbn = edbntoptrain(edbn, train_x, opts, train_y);

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, train_x, train_y);
fprintf('Scored: %2.2f\n', (1-er)*100);

%% Show the EDBN in action
spike_list = live_edbn(edbn, test_x(1, :), opts);
output_idxs = (spike_list.layers == numel(edbn.sizes));

figure(2); clf;
hist(spike_list.addrs(output_idxs) - 1, 0:edbn.sizes(end));
title('Label Layer Classification Spikes');
%% Export to xml to load into JSpikeStack
edbntoxml(edbn, opts, 'mnist_edbn');