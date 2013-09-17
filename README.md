# EDBN
====

EDBN is an event-based deep learning architecture for Matlab, first published in "Real-Time Classification and Sensor Fusion with a Spiking Deep Belief Network" by Peter O'Connor, Daniel Neil, Shih-Chii Liu, Tobi Delbruck and Michael Pfeiffer.  The purpose of this code is to provide a prototype example of the completed algorithms presented in the paper which can be modified, learned, or extended.

The original paper can be found at:
[http://www.frontiersin.org/neuromorphic%20engineering/10.3389/fnins.2013.00178/abstract](http://www.frontiersin.org/neuromorphic%20engineering/10.3389/fnins.2013.00178/abstract)

### Features

* Fast vectorized implementation.

* Selection of add-ons to improve accuracy in select domains: persistent contrastive divergence, fast weights, sparsity and selectivity, decay, momentum, temperature, and variable gibbs steps.

* Small filecount for easy modifications and extensions.

### Example

```matlab
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
% Use supervised training on the top layer
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
%% Export to xml
edbntoxml(edbn, opts, 'mnist_edbn');
```

### Installation

Unzip the repo and navigate to it within Matlab.  That's it.  If you'd like to test the installation, run the following matlab file:
```matlab
example.m
```
This will train a network on the MNIST handwritten digit database to greater than 92% accuracy, then run it on a spiking neuron network.  The source has been tested successfully with Matlab R2012a and later.

### File overview
The main files are referenced by the [example](example.m) file, which is compromised of the following execution order:
* example.m - runs an example
* edbnsetup.m - initializes the network and load defaults
* edbntrain.m - performs unsupervised training of the network
* erbmtrain.m - trains a single RBM layer in the DBN
* siegert.m - calculates the output spike rate of an input rate and input weights for LIF neurons
* erbmup.m / erbmdown.m - propagates rate-based activations up or down through LIF neurons
* edbntoptrain.m - performs supervised training by concatenating the top layer to the top-2 layer, and jointly training a (top-2, top)<->(top-1) RBM, then unrolling again.  
* edbnclean.m - cleans out all the temporary activations to save a minimum-size EDBN file
* edbntoxml.m - creates a base64-encoded representation of the network
* live_edbn.m - run the weights on an actual spiking network of neurons

The source also contains two .mat files which contain matlab structures:
* edbn_95.52.mat is a spiking network that achieves 95.52% recognition accuracy on the MNIST database of handwritten digits
* mnist_uint8.mat contains the mnsit network in matlab uint8 format, courtesy of the [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) and ultimately Yann LeCun.

Lastly, the source file that created the 95.52% accurate MNIST network is generated with [good_train.m](good_train.m)

### Questions
Please feel free to reach out here if you have any questions or difficulties.  I'm happy to help guide!