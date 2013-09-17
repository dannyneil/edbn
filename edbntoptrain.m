function edbn = edbntoptrain(edbn, data, opts, y)
    % Train the top layer by concatenating the top layer to a lower layer
    % and jointly training the set.
    orig_data = data;
    
    % Find merged layer and output layer
    merge_l = numel(edbn.erbm) - 1;
    out_l   = numel(edbn.erbm);
    
    % Build composite layer to train output; take top layer and place it
    % next to top-2 to joint train the layers: [top-2 top] <-> [top-1]
    top     = edbn.erbm{out_l};     
    top.W   = [edbn.erbm{merge_l}.W   edbn.erbm{out_l}.W'  ];
    top.vW  = [edbn.erbm{merge_l}.vW  edbn.erbm{out_l}.vW' ];
    top.FW  = [edbn.erbm{merge_l}.FW  edbn.erbm{out_l}.FW' ];
    top.vFW = [edbn.erbm{merge_l}.vFW edbn.erbm{out_l}.vFW'];
    top.v1  = [edbn.erbm{merge_l}.v1  edbn.erbm{out_l}.h1  ];
    top.v2  = [edbn.erbm{merge_l}.v2  edbn.erbm{out_l}.h2  ];
    top.b   = [edbn.erbm{merge_l}.b;  edbn.erbm{out_l}.c   ];
    top.vb  = [edbn.erbm{merge_l}.vb; edbn.erbm{out_l}.vc  ];
    top.h1  =  edbn.erbm{merge_l}.h1;
    top.h2  =  edbn.erbm{merge_l}.h2;
    top.c   =  edbn.erbm{merge_l}.c;
    top.vc  =  edbn.erbm{merge_l}.vc;
    
    % Train the composite layer
    if(opts.reup)
        opts.epoch_loops = opts.numepochs;
        opts.numepochs = 1;
    else
        opts.epoch_loops = 1;
    end    
    
    for i = 0:opts.epoch_loops-1
        opts.ep_st = i;
        % Pass the data up
        for i = 1:numel(edbn.erbm)-2    
            data = erbmup(edbn.erbm{i}, data)';
        end
        top = erbmtrain(top, [data y], opts); 
        data = orig_data;
    end
    
    % Slice the top layer from the composite layer
    slice = edbn.sizes(merge_l)+1 : edbn.sizes(merge_l)+edbn.sizes(out_l+1);
    edbn.erbm{out_l}.W   = top.W(:, slice)';
    edbn.erbm{out_l}.vW  = top.vW(:, slice)';
    edbn.erbm{out_l}.FW  = top.FW(:, slice)';
    edbn.erbm{out_l}.vFW = top.vFW(:, slice)';    
    edbn.erbm{out_l}.h1  = top.v1(:, slice);
    edbn.erbm{out_l}.h2  = top.v2(:, slice);      
    edbn.erbm{out_l}.v1  = top.h1( :, : );
    edbn.erbm{out_l}.v2  = top.h2( :, : );  
    edbn.erbm{out_l}.b   = top.c( : );
    edbn.erbm{out_l}.vb  = top.vc( : );
    edbn.erbm{out_l}.c   = top.b(slice);
    edbn.erbm{out_l}.vc  = top.vb(slice);        
        
    % Slice off the composite layer into its original layer
    slice = 1 : edbn.sizes(merge_l);
    edbn.erbm{merge_l}.W   = top.W (:, slice);
    edbn.erbm{merge_l}.vW  = top.vW(:, slice);
    edbn.erbm{merge_l}.FW  = top.FW (:, slice);
    edbn.erbm{merge_l}.vFW = top.vFW(:, slice);        
    edbn.erbm{merge_l}.h1  = top.h1( :, : );
    edbn.erbm{merge_l}.h2  = top.h2( :, : );
    edbn.erbm{merge_l}.v1  = top.v1(:, slice);
    edbn.erbm{merge_l}.v2  = top.v2(:, slice);    
    edbn.erbm{merge_l}.b   = top.b(slice);
    edbn.erbm{merge_l}.vb  = top.vb(slice);
    edbn.erbm{merge_l}.c   = top.c;
    edbn.erbm{merge_l}.vc  = top.vc;    
    
end