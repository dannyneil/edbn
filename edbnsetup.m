function [edbn, opts] = edbnsetup(edbn, opts)
    assert (numel(edbn.sizes) > 1, 'Sizes must be given for all layers, including input and output.');

    % Set defaults
    if ~isfield(opts,'alpha'),     opts.alpha     =      1; end
    if ~isfield(opts,'decay'),     opts.decay     = 0.0001; end
    if ~isfield(opts,'momentum'),  opts.momentum  =    0.0; end
    if ~isfield(opts,'temp'),      opts.temp      =  0.005; end
    if ~isfield(opts,'tau_m'),     opts.tau_m     =    5.0; end
    if ~isfield(opts,'tau_s'),     opts.tau_s     =  0.001; end
    if ~isfield(opts,'t_ref'),     opts.t_ref     =  0.002; end
    if ~isfield(opts,'v_thr'),     opts.v_thr     =  0.005; end
    if ~isfield(opts,'f_infl'),    opts.f_infl    =      1; end
    if ~isfield(opts,'f_decay'),   opts.f_decay   =   0.05; end
    if ~isfield(opts,'f_alpha'),   opts.f_alpha   =      5; end
    if ~isfield(opts,'pcd'),       opts.pcd       =      1; end
    if ~isfield(opts,'sp'),        opts.sp        =    0.1; end
    if ~isfield(opts,'sp_infl'),   opts.sp_infl   =    0.2; end
    if ~isfield(opts,'ngibbs'),    opts.ngibbs    =      2; end
    if ~isfield(opts,'initscl'),   opts.initscl   =   0.01; end
    if ~isfield(opts,'batchsize'), opts.batchsize =     50; end
    if ~isfield(opts,'reup'),      opts.reup      =      1; end
    if ~isfield(opts,'wtreset'),   opts.wtreset   =      1; end

    for u = 1 : numel(edbn.sizes) - 1
        % Set up constants for learning
        edbn.erbm{u}.alpha    = opts.alpha;
        edbn.erbm{u}.decay    = opts.decay;
        edbn.erbm{u}.momentum = opts.momentum;
        edbn.erbm{u}.f_infl   = opts.f_infl;
        edbn.erbm{u}.f_decay  = opts.f_decay;
        edbn.erbm{u}.f_alpha  = opts.f_alpha;
        edbn.erbm{u}.pcd      = opts.pcd;
        edbn.erbm{u}.sp       = opts.sp;
        edbn.erbm{u}.sp_infl  = opts.sp_infl;

        % Set up constants for the Siegert formula
        edbn.erbm{u}.sieg.tau_m = opts.tau_m;
        edbn.erbm{u}.sieg.tau_s = opts.tau_s;
        edbn.erbm{u}.sieg.t_ref = opts.t_ref;
        edbn.erbm{u}.sieg.v_thr = opts.v_thr;
        edbn.erbm{u}.sieg.temp  = opts.temp;

        % Weights, lower bias, and upper bias
        if(opts.wtreset)
            edbn.erbm{u}.W  = opts.initscl*randn(edbn.sizes(u+1), edbn.sizes(u));        
            edbn.erbm{u}.b  = zeros(edbn.sizes(u  ), 1);
            edbn.erbm{u}.c  = zeros(edbn.sizes(u+1), 1);
        end
        
        % Delta values
        edbn.erbm{u}.vW  = zeros(edbn.sizes(u+1), edbn.sizes(u));        
        edbn.erbm{u}.vb  = zeros(edbn.sizes(u  ), 1);
        edbn.erbm{u}.vc  = zeros(edbn.sizes(u+1), 1);                
        
        % Temporary storage for data and model intermediates as well as fast weights
        edbn.erbm{u}.v1  = zeros(opts.batchsize, edbn.sizes(u));
        edbn.erbm{u}.h1  = zeros(opts.batchsize, edbn.sizes(u+1));
        edbn.erbm{u}.v2  = zeros(opts.batchsize, edbn.sizes(u));
        edbn.erbm{u}.h2  = zeros(opts.batchsize, edbn.sizes(u+1));
        edbn.erbm{u}.FW  = zeros(edbn.sizes(u+1), edbn.sizes(u));
        edbn.erbm{u}.vFW = zeros(edbn.sizes(u+1), edbn.sizes(u));
    end

end

