% idle waiting
action(train(ID), wait, 0) :- train(ID).
action(train(ID), wait, 0..Dep-1) :- start(ID,(_,_),Dep,Dir), Dep > 0.

% spawning action
action(train(ID), move_forward, Dep) :- start(ID, (X,Y), Dep, Dir), Dep > 0.
action(train(ID), move_forward, 1) :- start(ID, (X,Y), 0, Dir).

% fact format: position(train ID, coordinates, facing direction, timestep)
% init position
position(ID, (X,Y), Dir, Dep+1) :- start(ID, (X,Y), Dep, Dir), Dep >0.
position(ID, (X,Y), Dir, 2) :- start(ID, (X,Y), 0, Dir).

% latest arrival time of train
max_end_time(ID,E) :- end(ID,(_,_),E).

% arrival time of train
end_time(ID, T) :- position(ID, (X,Y), _, T), end(ID, (X,Y), _).

% generate new position after action
% restrictions in rule body instead of constraints later, to limit grounding of recursive rule
% optimizations: only when new cell exists, new cell is a track, old cell allows movement
position(ID, (X+DX,Y+DY), New_Dir, T) :- position(ID, (X,Y), Old_Dir, T-1), action(train(ID), M, T-1), movement(M, Old_Dir, (DX,DY), New_Dir), 
                                        M != wait, not forced_turn((X,Y), Old_Dir,_), cell((X+DX,Y+DY),TrackID), TrackID != 0,
                                        cell((X,Y), TrackID_old), possible_movement(TrackID_old, Old_Dir, M).

position(ID, (X+DX,Y+DY), New_Dir, T) :- position(ID, (X,Y), Old_Dir, T-1), action(train(ID), move_forward, T-1), movement(M, Old_Dir, (DX,DY), New_Dir), 
                                        forced_turn((X,Y), Old_Dir, M), cell((X+DX,Y+DY),TrackID), TrackID != 0.

position(ID, (X,Y), Dir, T) :- position(ID, (X,Y), Dir, T-1), action(train(ID), wait, T-1).

% want a rule that choses exactly one of each possible movements per AS on the TrackID of the current position
{action(train(ID), Action, T) : possible_movement(TrackID, Dir, Action)}=1 :- 
                    position(ID, (X,Y), Dir, T), cell((X,Y), TrackID), not end(ID, (X,Y), _), max_end_time(ID,E), T < (E+1).                 

% minimize time
%#minimize { T : end_time(ID, T) }.

% trains may not swap
:- position(ID1, (X1,Y1), _, T), position(ID2, (X2,Y2), _, T), position(ID1, (X2,Y2), _, T+1), position(ID2, (X1,Y1), _, T+1), ID1 != ID2.

% no two trains on the same cell
:- position(ID1, (X,Y), Dir1, T), position(ID2, (X,Y), Dir2, T), ID1 != ID2.

% every train has to reach end
:- end(ID, (X,Y), _), not position(ID, (X,Y), _, _).

#show action/3.
%#show position/4.
%#show possible_movement/3.
