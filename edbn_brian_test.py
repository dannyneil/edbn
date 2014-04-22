from brian import *
from numpy import *
from functools import partial
import scipy.io as sio
import multiprocessing

## Quick classification demonstration using BRIAN's LIF neurons
##   To run: python ./edbn_brian_test.py
##   Requires BRIAN, numpy, and scipy

# Load MNIST data
def load_data(matfile):
    dataset = sio.loadmat('mnist_uint8.mat')
    test_x = dataset['test_x'] / 255.0 * 0.2
    test_y = dataset['test_y'] / 255.0 * 0.2
    return test_x, test_y

# Thanks to Evangelos Stromatias for this Matlab->Python interface function:
def loadMatEDBNDescription(matfile):
    '''
        This function opens a MAT description of a trainned Event-Based
        Deep Belief Network and returns the number of neurons in each layer of the network 
        and a list of NUMPY arrays of the synaptic weights of each layer.
    '''
    edbn = sio.loadmat(matfile)    
    # Get size and dimensions of layers
    edbnTopology =  edbn['edbn']['sizes'][0][0][0] #number of layers and neurons
    # Layer weights are saved at this index:
    _WEIGHTStr = 10     
    # Get weights for each layer
    weightList = []
    for l in range(edbnTopology.size-1):
        weightList.append(edbn['edbn']['erbm'][0][0][0][l][0][0][_WEIGHTStr])
    return edbnTopology, weightList


# Core of this script: initialize a net, load a digit, and test its results
def run_digit(test_data, weightlist, topology, params):
    reinit_default_clock()
    clear(True)
    test_x = test_data[0]
    test_y = test_data[1]
    pops = []
    conns = []
    # Build populations
    for (layer_num, layer_size) in enumerate(edbn_topology):
        if layer_num == 0:
            pops.append(PoissonGroup(layer_size))
        else:
            pops.append(NeuronGroup(layer_size, model=params['eqs'], threshold=params['v_th'], reset=params['v_r']))
    # Build connections
    for (layer_num, weights) in enumerate(weightlist):
        c = Connection(pops[layer_num], pops[layer_num+1], 'v', weight=weights.T*1000*mV)
        conns.append(c)

    # Set a spike rate of 0.2 * 30 for a fully on digit
    pops[0].rate = test_x * 30.0
    # Track the output layer spikes
    output_spikes = SpikeCounter(pops[-1])
    # Run for one second, approximately 1000 spikes
    run(1.0*second)
    # Get digit guess and correct answer
    guessed_digit = np.argmax(output_spikes.count)
    correct_digit = np.argmax(test_y)
    # Return true if correct

    return guessed_digit == correct_digit

if __name__ == "__main__":
    # How many digits to test
    num_to_test = 12
    # Set parameters
    params = {}
    params['tau_m'] = 5000 * ms # membrane time constant
    params['v_r'] = 0 * mV # reset potential
    params['v_th'] = 1000 * mV # threshold potential
    params['eqs'] = '''
        dv/dt = -v/params['tau_m'] : volt
        '''
    # Load network
    edbn_topology, weightlist = loadMatEDBNDescription('edbn_95.52.mat')
    # Load data
    test_x, test_y = load_data('mnist_uint8.mat')
    # Build partial function to pass to the mapper
    partial_run_digit = partial(run_digit, topology=edbn_topology, weightlist=weightlist, params=params)
    # Initialize a multiprocessing pool
    pool = multiprocessing.Pool(4)
    # Distribute the test data into a set of tuples with the label
    test_data = [(test_x[idx, :], test_y[idx,:]) for idx in range(num_to_test) ]
    # Map the function over the pool
    results = pool.map(partial_run_digit, test_data)
    # Spit out results
    print "%i correct answers in %i trials for %.2f accuracy." % (np.sum(results), num_to_test, float(np.sum(results))/num_to_test*100)
