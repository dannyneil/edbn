function erbm = erbmtrain(erbm, x, opts)
    % Check the inputs
    assert(isfloat(x), 'Data must be a float.');
    m = size(x, 1);
    numbatches = m / opts.batchsize;    
    assert(rem(numbatches, 1) == 0, 'Numbatches is not an integer.');

    % Preallocate
    linsel  = linspace(0, 1, opts.batchsize);
    linsp   = linspace(0, 1, size(erbm.W,1));
    
    % Loop and train
    for ep = 1 : opts.numepochs
        kk = randperm(m);
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            % Obtain data sample
            erbm.v1 = batch;
            erbm.h1 = siegert(erbm.v1', erbm.W , erbm.sieg)';
            if(~erbm.pcd)
                erbm.h2 = erbm.h1;
            end                
            
            % Obtain model sample, using fast weights to explore quickly
            for g = 1:opts.ngibbs
                erbm.v2 = siegert(erbm.h2', erbm.W' + ...
                    erbm.f_infl * erbm.FW', erbm.sieg)';
                erbm.h2 = siegert(erbm.v2', erbm.W  + ...
                    erbm.f_infl * erbm.FW , erbm.sieg)';
            end
            
            % Sparsify; see Goh, Thome, Cord
            [~,  ixsp]    = sort(erbm.h1, 2);
            [~,  ordersp] = sort(ixsp   , 2);
            ranksp        = linsp(ordersp);
            h1sp          = ranksp.^(1/erbm.sp-1);                               
            [~, ixsel]    = sort(h1sp, 1);
            [~, ordersel] = sort(ixsel, 1);
            ranksel       = linsel(ordersel);
			h1sp          = ranksel.^(1/erbm.sp-1);                    
			erbm.h1 = erbm.sp_infl * h1sp + (1 - erbm.sp_infl) * erbm.h1;
            
            % Calculate activation correlations
            c1 = erbm.h1' * erbm.v1;
            c2 = erbm.h2' * erbm.v2;

            % Update fast weights and biases
            erbm.vFW = erbm.f_alpha / opts.batchsize * (c1 - c2);
            dW       = erbm.alpha   / opts.batchsize * (c1 - c2);
            db       = erbm.alpha   / opts.batchsize * sum(erbm.v1 - erbm.v2)';
            dc       = erbm.alpha   / opts.batchsize * sum(erbm.h1 - erbm.h2)';

            % Incorporate decay
            erbm.FW = (1 - erbm.f_decay) * erbm.FW + erbm.vFW;
            dW = dW - erbm.decay * erbm.alpha * erbm.W;
            db = db - erbm.decay * erbm.alpha * erbm.b;
            dc = dc - erbm.decay * erbm.alpha * erbm.c;
            
            % Incorporate momentum
            erbm.vW = opts.momentum * erbm.vW + dW;
            erbm.vb = opts.momentum * erbm.vb + db;
            erbm.vc = opts.momentum * erbm.vc + dc;
            
            % Update final values
            erbm.W = erbm.W + erbm.vW;
            erbm.b = erbm.b + erbm.vb;
            erbm.c = erbm.c + erbm.vc;
        end
        
        % Inform the user
        fprintf('Epoch %i: mean error: %1.5f.\n', ...
            ep+opts.ep_st, mean(abs(sum(erbm.v2) - sum(erbm.v1))) / opts.batchsize);
    end
end