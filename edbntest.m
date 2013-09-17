function [er, bad] = edbntest(edbn, x, y)
    % Pass activations up
    for i = 1:numel(edbn.erbm)
        x = erbmup(edbn.erbm{i}, x)';
    end
    
    % Get winners from top row labels
    [~,    found] = max(x,[],2);
    [~, expected] = max(y,[],2);
    bad = find(found ~= expected);    
    er = numel(bad) / size(y, 1);
end
