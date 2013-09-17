function x = erbmup(erbm, x)
    %erbm.sieg.temp = 0;
    x = siegert(x', erbm.W, erbm.sieg);
end
