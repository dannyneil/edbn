function [spike_list, edbn, opts] = live_edbn(edbn, x, opts)
% Set defaults
if ~isfield(edbn,'fw_conns'), edbn.fw_conns=num2cell([2:numel(edbn.sizes) 0]); end;
if ~isfield(edbn,'fb_conns'), edbn.fb_conns=num2cell([0 1:numel(edbn.sizes)-1]); end;
if ~isfield(edbn,'fw_erbm'),  edbn.fw_erbm=[1:numel(edbn.erbm) 0]; end

if ~isfield(opts,'recreate'),   opts.recreate    =      1; end;    
if ~isfield(opts,'timespan'),   opts.timespan    =      4; end;
if ~isfield(opts,'numspikes'),  opts.numspikes   =   2000; end;
if ~isfield(opts,'delay'),      opts.delay       =  0.001; end;
if ~isfield(opts,'show_dt'),    opts.show_dt     =  0.010; end;
if ~isfield(opts,'vis_tau'),    opts.vis_tau     =   0.05; end;
if ~isfield(opts,'makespikes'), opts.makespikes  =      1; end;
if ~isfield(opts,'makevisdim'), opts.makevisdim  =      1; end;
if ~isfield(opts,'vis_handle'), opts.vis_handle  = figure; end;
if ~isfield(opts,'vis_layers'), opts.vis_layers  = 1:numel(edbn.sizes); end;

if ~isfield(opts,'ff'), opts.ff = [ones(1, numel(edbn.sizes)-1) 0]; end;
if ~isfield(opts,'fb'), opts.fb = zeros(1, numel(edbn.sizes)); end;

% Create reconstruction layer as necessary
if(opts.recreate) 
    edbn.sizes = [edbn.sizes edbn.sizes(1)];
    edbn.erbm  = [edbn.erbm   edbn.erbm(1)];
    edbn.fb_conns{2} = numel(edbn.sizes);
    edbn.fw_conns = [edbn.fw_conns edbn.fw_conns{1}];
    edbn.fb_conns = [edbn.fb_conns 0];
    opts.ff = [opts.ff 0];
    opts.fb(2) = 1;
    opts.fb = [opts.fb 0];
    edbn.fw_erbm = [edbn.fw_erbm numel(edbn.erbm)];
    opts.vis_layers = [opts.vis_layers numel(edbn.sizes)];
end

% Create spikes proportional to intensity and assign a random time
if(opts.makespikes)
    inp.addr  = randsample(numel(x), opts.numspikes, true, x(:))';
    inp.times = sort(rand(1,opts.numspikes)) .* opts.timespan;
    x = inp;
end

% Initialize neurons
layers     = numel(edbn.sizes);
mem        = cell(1, layers);
mem_time   = cell(1, layers);
refrac_end = cell(1, layers);
for i = 1:layers
    mem{i}        = zeros(1, edbn.sizes(i));    
    mem_time{i}   = 0;    
    refrac_end{i} = zeros(1, edbn.sizes(i));
end

% Initialize spike queues
queue.times = x.times;
queue.layers = 1 * ones(1, numel(x.addr));
queue.addrs = num2cell(x.addr);

% Build show dimensions
if(opts.makevisdim)
    for i = 1:layers
        factors = factor(edbn.sizes(i));
        opts.show_dims{i} = [prod(factors(2:2:end)) prod(factors(1:2:end))];
    end
end

% Build plotspace
figure(opts.vis_handle); clf;
last_spiked = cell(1, layers);
image_handle = cell(1, layers);
for i=1:numel(edbn.sizes)
    % Create storage for visualization
    last_spiked{i} = zeros(1, edbn.sizes(i));
    
    if(any(i==opts.vis_layers))            
        % Plot and store a handle
        subplot(1, numel(opts.vis_layers), find(i==opts.vis_layers));        
        image(reshape(last_spiked{i}, opts.show_dims{i})', 'CDataMapping','scaled');
        colormap bone; axis image; axis off;
        image_handle{i} = get(gca,'Children');
    end
end

% Run the event-based network
tic;
idx = 1;
last_show = 0;
while(idx < size(queue.times, 2))
    % Pull out spikes to process
    curr_time  = queue.times(idx);
    from_addr  = queue.addrs{idx};
    from_layer = queue.layers(idx);
    
    % Update display
    last_spiked{from_layer}(from_addr) = last_spiked{from_layer}(from_addr) + 1;
    
    % Process forward and backward connections
    for direction=1:2
        if(direction == 1), conns = opts.ff(from_layer) .* edbn.fw_conns{from_layer}; end
        if(direction == 2), conns = opts.fb(from_layer) .* edbn.fb_conns{from_layer}; end
        
        % Skip if nothing to do
        if(~any(conns)), continue; end
        
        for l_idx = 1:numel(conns)
            layer = conns(l_idx);
        
            % Decay
            time_gap   = curr_time - mem_time{layer};
            decayfac   = exp(-time_gap / opts.tau_m);
            mem{layer} = mem{layer} .* decayfac;

            % Add Impulse
            not_refrac = curr_time > refrac_end{layer};
            if(direction == 1)
                impulse = edbn.erbm{edbn.fw_erbm(from_layer)}.W(:, from_addr)';
            else
                impulse = edbn.erbm{edbn.fw_erbm(layer)}.W(from_addr, :);
            end
            mem{layer} = mem{layer} + sum(bsxfun(@times, not_refrac, impulse), 1);

            % Store new 'old' time
            mem_time{layer} = curr_time;

            % Check for firing; reset potential and store refrac
            firings = find(mem{layer} > opts.v_thr);
            mem{layer}(firings) = 0;
            refrac_end{layer}(firings) = curr_time + opts.t_ref;

            % Update queues
            if(numel(firings > 0))
                insert_idx = find(curr_time + opts.delay < queue.times, 1, 'first');
                if(isempty(insert_idx))
                    insert_idx = size(queue, 2);
                end
                queue.times = horzcat(queue.times(1:insert_idx-1), ...
                                      curr_time + opts.delay, ...
                                      queue.times(insert_idx:end));
                queue.layers = horzcat(queue.layers(1:insert_idx-1), ...
                                       layer, ...
                                       queue.layers(insert_idx:end));            
                queue.addrs = horzcat(queue.addrs(1:insert_idx-1), ...
                                      {firings}, ...
                                      queue.addrs(insert_idx:end));
            end
        end
    end   
    
    % Display if desired
    if(curr_time - last_show > opts.show_dt)
        for i = 1:numel(edbn.sizes)
            last_spiked{i} = last_spiked{i} .* exp(-opts.show_dt / opts.vis_tau);
            if(any(i==opts.vis_layers))
                set(image_handle{i}, 'CData', reshape(last_spiked{i}, opts.show_dims{i})');
            end
        end
        pause(0.01);
        last_show = curr_time;
    end
    
    % Pick next time
    idx = idx + 1;
end
% Report finish and the timing
fprintf('Completed %i input spikes occurring over %2.2f seconds, in %2.3f seconds of real time.\n', ...
    opts.numspikes, opts.timespan, toc);

% De-parallelize output
spike_list.addrs  = cell2mat(queue.addrs);
numrepeats = cellfun(@(x)size(x,2),queue.addrs);
spike_list.times  = cell2mat(arrayfun(@(time, repeats) ones(1,repeats)*time, queue.times, numrepeats, 'UniformOutput', 0));
spike_list.layers = cell2mat(arrayfun(@(layer, repeats) ones(1,repeats)*layer, queue.layers, numrepeats, 'UniformOutput', 0));
