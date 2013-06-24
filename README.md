# EDBN
====

EDBN is an event-based deep learning architecture for Matlab.  Once the paper review has been completed, the source code will be published here for use.

### Features

* Fast vectorized implementation.

* Selection of add-ons: persistent contrastive divergence, fast weights, sparsity and selectivity, decay, momentum, temperature, and variable gibbs steps.

* Small filecount for easy modifications and extensions.

### Example

```matlab
%% Load paths
EDBNinit;

%% Load data
load mnist_uint8;

train_x = double(train_x) / 255 * 0.2;
test_x  = double(test_x)  / 255 * 0.2;
train_y = double(train_y) * 0.2;
test_y  = double(test_y)  * 0.2;

%% Train network
rand('seed', 42);
edbn.sizes = [784 500 500 10];
opts.batchsize = 50;
[edbn, opts] = edbnsetup(edbn,opts);

opts.momentum = 0.8; 
opts.numepochs = 3;
edbn = edbntrain(edbn, train_x, opts);
edbn = edbntoptrain(edbn, train_x, opts, train_y);

figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, train_x, train_y);
fprintf('Scored: %2.2f\n', (1-er)*100);
```

### Installation

Unzip the repo and navigate to it within Matlab.  Then, run:
```matlab
EDBNinit
```
to set up the paths.  Finally, test the installation by executing the example above.
