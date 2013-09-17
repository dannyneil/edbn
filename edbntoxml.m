function edbntoxml(edbn, opts, name)

% Load defaults
if ~isfield(opts,'makevisdim'), opts.makevisdim =     1; end;
if ~isfield(edbn,'fw_conns'), edbn.fw_conns=num2cell([2:numel(edbn.sizes) 0]); end;
if ~isfield(edbn,'fw_erbm'),  edbn.fw_erbm=[1:numel(edbn.erbm) 0]; end

% Clean out all temporary variables
edbn = edbnclean(edbn);

% Set up metadata
net.name    = name;
net.type    = 'edbn';
net.notes   = '';
net.nLayers = numel(edbn.sizes);
net.param   = edbn.erbm{1}.sieg;
net.dob     = datestr(now);

% Build show dimensions
if(opts.makevisdim)
    for i = 1:numel(edbn.sizes)
        factors = factor(edbn.sizes(i));
        opts.show_dims{i} = [prod(factors(2:2:end)) prod(factors(1:2:end))];
    end
end

% Encode data in base64 for export
encode = @(x)base64encode(typecast(single(x(:)),'uint8'),'matlab');

% Build layers with all parameters
net.Layer=cell(1, numel(edbn.sizes));
for i = 1:numel(edbn.sizes)
    lay.index  = i-1;
    lay.nUnits = edbn.sizes(i);
    lay.dimx   = opts.show_dims{i}(1);
    lay.dimy   = opts.show_dims{i}(2);
    lay.thresh.Attributes.dt = 'base64-single';

    if (edbn.fw_erbm(i) ~= 0)
        lay.targ   = i;
        lay.param  = edbn.erbm{edbn.fw_erbm(i)}.sieg;            
        lay.W.Attributes.dt = 'base64-single';
        lay.W.Text = encode(edbn.erbm{edbn.fw_erbm(i)}.W);
        lay.thresh.Text=encode(repmat(edbn.erbm{edbn.fw_erbm(i)}.sieg.v_thr, 1, lay.nUnits));        
    else
        lay.targ = '-';        
        lay.param = edbn.erbm{end}.sieg;
        lay.W.Attributes.dt = 'base64-single';
        lay.W.Text = encode([]);                
        lay.thresh.Text=encode(repmat(edbn.erbm{end}.sieg.v_thr, 1, lay.nUnits));                
    end

    net.Layer{i} = lay;
end

% Encapsulate
outnet.Network = net;
% Write XML
struct2xml(outnet, strcat(name, '.xml'));