# file: eval_helper.jl
# author: mingyuw@stanford.edu



function pass_behavior(trajectory)
    # input: a state trajectory of two agents [x1, y1, v1, theta1, x2, y2, v2, theta2]
    # output: 1 if player 1 passes ahead of player 2, 2 otherwise
    for t = 1:size(trajectory)[1] - 1
        prev1 = trajectory[t,1:2]
        prev2 = trajectory[t,5:6]
        current1 = trajectory[t+1,1:2]
        current2 = trajectory[t+1,5:6]

        if prev1[1] < prev2[1] && current1[1] > current2[1]
            if(current1[2] > current2[2])
                return 1
            else
                return 2
            end
        end
    end
    return 3
end
