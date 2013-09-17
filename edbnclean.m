function edbn = edbn_clean(edbn)

for l=1:numel(edbn.erbm);
    % Wipe the extra stuff
    edbn.erbm{l}.vW  = [];
    edbn.erbm{l}.vb  = [];
    edbn.erbm{l}.vc  = [];
    edbn.erbm{l}.FW  = [];
    edbn.erbm{l}.vFW = [];
    edbn.erbm{l}.h1  = [];
    edbn.erbm{l}.h2  = [];
    edbn.erbm{l}.v1  = [];
    edbn.erbm{l}.v2  = [];
end
