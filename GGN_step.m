%% original program from https://stackoverflow.com/questions/1903104/generalized-gaussian-noise-generator-in-matlab
function GGN_matrix = GGN_step(alpha, beta, matrix_x, matrix_y)
GGN_matrix = zeros(matrix_x, matrix_y);
for x_index=1:matrix_x
    for y_index=1:matrix_y
        v = 10/(alpha^2)* rand(1) - 5/(alpha^2);
        while(p(v, alpha, beta) < rand(1))
            v = 10/alpha* rand(1) - 5/alpha;
        end
        GGN_matrix(x_index,y_index) = v;
    end
end

function pval = p(v, alpha, beta)
pval = (alpha*beta/(2*gamma(1/beta))) * exp(-(alpha*abs(v )).^beta );



