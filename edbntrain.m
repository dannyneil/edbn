function edbn = edbntrain(edbn, data, opts)
    % Initialize the training count
    opts.ep_st = 0;
    
    % Expect the final two layers to be top-trained
    for l = 1:numel(edbn.erbm) - 2
        edbn.erbm{l} = erbmtrain(edbn.erbm{l}, data, opts);
        data = erbmup(edbn.erbm{l}, data)';
    end 
end