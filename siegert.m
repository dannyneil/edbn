function y=siegert(x,w,opts)
% Compute the ouput of an array of Siegert Neurons.  Siegert Neurons have
% outputs that approximate the mean firing rate of leaky integrate-and-fire
% neurons with the same parameters
%
% y=siegert(x,w,P)
%
% Example Usage:
%  nInputs=5;
%  nSamples=100;
%  nUnits=3;
%  x=bsxfun(@plus,rand(nInputs,nSamples),linspace(0,500,nSamples));
%  w=.4*randn(nUnits,nInputs)+.8;
%  P=struct; opts.Vth=2; opts.tref=.01;
%  plot(siegert(x,w,P)'); xlabel 'input rate';ylabel 'output rate'; 
%  
%
% Input     Dim             Description
% x         [nIn,nSamples]  set of input vectors representing Poisson firing rates of inputs
% w         [nOut,nIn]      weight matrix
% P         -               structure identifying neuron parameters (see below).
%
% Ouput     Dims            Description
% y         [nOut,nSamples] set of output vectors representing Poisson output rate
%
% P is a structure array as fields of P
% Vrest         Resting Potential
% Vth           Threshold.  This can be a scalar or a vector with 1 element per unit.
% Vreset        Post-Spike Reset Potential
% taum          Membrane Time Constent
% tausyn        Synaptic response time constant.  LIF-equivalent synapic response is assumed to follow exp(x/tausyn)-exp(-x/(2*tausyn)) 
% tref          Absolute refractory Time
% normalize     Normalize output to the maximum possible firing rate (1/tref) 
% nPoints       Number of points in integral approximation
% polyapprox    Boolean indicating whether to approximate with a polynomial.  Generally it's best to leave this true.
% imagrownadult Don't waste time checking input to ensure that all inputs are positive.  The caller can do this. 
% 
% See Florian Jug's poster explaining the Siegert neuron at:
% http://www.cadmo.ethz.ch/as/people/members/fjug/personal_home/posters/2012_SiegertPoster.pdf
%
% Peter 
% oconnorp ..at.. ethz ..dot.. ch
%


%% Process Inputs
% Add noise
x = max(x + opts.temp * randn(size(x)), 0);

% Check inputs
k=sqrt( opts.tau_s / opts.tau_m);
assert( size(w,2) == size(x,1),'Weight matrix columns must equal input matrix rows.');
assert( k < 1,'Tau_m must be greater than tau_s.');
assert( all(x(:) >= 0),'Input rate cannot be negative.');
%% Perform computation!
% State constants
opts.Vrest = 0;
opts.Vreset = 0;
opts.nPoints = 3;
opts.polyapprox = 1;

% Compute intermediates
Y = opts.Vrest + opts.tau_m * w * x;
G = sqrt(opts.tau_m * w.^2 * x / 2);

% Load constant
persistent SIEGERT_GAM;
if(isempty(SIEGERT_GAM))
    SIEGERT_GAM = abs(zeta(0.5));
end
gam = SIEGERT_GAM;

% Approximate the integral
tmp = reshape(bsxfun(@plus, bsxfun(@times, ...
                                   meshgrid(1:opts.nPoints, 0:size(w,1)-1), ...
                                   opts.v_thr(:)-opts.Vreset(:))/(opts.nPoints-1), ...
                            opts.Vreset), ...
             [size(w,1) 1 opts.nPoints]);
u = bsxfun(@plus, (k * gam) * G, tmp);

du = (opts.v_thr - opts.Vreset) / (opts.nPoints - 1);
YminusuoverGroot2 = bsxfun(@rdivide, bsxfun(@minus, Y, u) , G * sqrt(2));

% Approximate the integral.  
%   Note: erfcx is a more numerically stable version of exp(x.^2)*erfc(x)
if opts.polyapprox
    z = erfcx(YminusuoverGroot2);
    % Apply Simpon's rule to approximate the integral. Generally, a good
    % approximation as the integration curve is close to a parabola.    
    integral = bsxfun(@times, (opts.v_thr(:) - opts.Vreset(:)) / 6, ...
        z(:,:,1) + 4 * z(:,:,2) + z(:,:,3) ); 
else
    integral = du * sum(erfcx(YminusuoverGroot2), 3); 
end

% Compute the output
y = 1 ./ (opts.t_ref + (opts.tau_m ./ G) .* (sqrt(pi / 2)) .* integral);

% Normalize to 1/t_ref
y = y * opts.t_ref; 