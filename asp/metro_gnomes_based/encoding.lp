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

%*% curves
curve(X,Y) :- cell((X,Y),72).
curve(X,Y) :- cell((X,Y),4608).
curve(X,Y) :- cell((X,Y),16386).
curve(X,Y) :- cell((X,Y),2064).

% symmetrical switches
symmetrical_switch(X,Y) :- cell((X,Y),20994).
symmetrical_switch(X,Y) :- cell((X,Y),16458).
symmetrical_switch(X,Y) :- cell((X,Y),2136).
symmetrical_switch(X,Y) :- cell((X,Y),6672).*%

% movement execution depending on action and direction 
% format: movement(action, old_dir, coordinate change, new_dir)
movement(move_left, n, (0,-1), w).
movement(move_left, e, (-1,0), n).
movement(move_left, s, (0,1), e).
movement(move_left, w, (1,0), s).

movement(move_right, n, (0,1), e).
movement(move_right, e, (1,0), s).
movement(move_right, s, (0,-1), w).
movement(move_right, w, (-1,0), n).

movement(move_forward, n, (-1,0), n).
movement(move_forward, e, (0,1), e).
movement(move_forward, s, (1,0), s).
movement(move_forward, w, (0,-1), w).

% designates cells on which going in Dir makes you turn with a forward move
% format: forced_turn(coordinates, old_dir, turn_dir)
forced_turn((X,Y), s, move_left) :- cell((X,Y),72).
forced_turn((X,Y), w, move_right) :- cell((X,Y),72).
forced_turn((X,Y), n, move_left) :- cell((X,Y),4608).
forced_turn((X,Y), e, move_right) :- cell((X,Y),4608).
forced_turn((X,Y), w, move_left) :- cell((X,Y),16386).
forced_turn((X,Y), n, move_right) :- cell((X,Y),16386).
forced_turn((X,Y), e, move_left) :- cell((X,Y),2064).
forced_turn((X,Y), s, move_right) :- cell((X,Y),2064).

forced_turn((X,Y), w, move_left) :- cell((X,Y),20994).
forced_turn((X,Y), e, move_right) :- cell((X,Y),20994).
forced_turn((X,Y), e, move_left) :- cell((X,Y),2136).
forced_turn((X,Y), w, move_right) :- cell((X,Y),2136).
forced_turn((X,Y), s, move_left) :- cell((X,Y),16458).
forced_turn((X,Y), n, move_right) :- cell((X,Y),16458).
forced_turn((X,Y), n, move_left) :- cell((X,Y),6672).
forced_turn((X,Y), s, move_right) :- cell((X,Y),6672).

forced_turn((X,Y), n, move_left) :- cell((X,Y),5633).
forced_turn((X,Y), n, move_right) :- cell((X,Y),17411).
forced_turn((X,Y), e, move_left) :- cell((X,Y),34864).
forced_turn((X,Y), e, move_right) :- cell((X,Y),37408).
forced_turn((X,Y), s, move_left) :- cell((X,Y),1097).
forced_turn((X,Y), s, move_right) :- cell((X,Y),3089).
forced_turn((X,Y), w, move_left) :- cell((X,Y),49186).
forced_turn((X,Y), w, move_right) :- cell((X,Y),32872).

% generate new position after action
position(ID, (X+DX,Y+DY), New_Dir, T) :- position(ID, (X,Y), Old_Dir, T-1), action(train(ID), M, T-1), movement(M, Old_Dir, (DX,DY), New_Dir), M != wait, not forced_turn((X,Y), Old_Dir,_).
position(ID, (X+DX,Y+DY), New_Dir, T) :- position(ID, (X,Y), Old_Dir, T-1), action(train(ID), move_forward, T-1), movement(M, Old_Dir, (DX,DY), New_Dir), forced_turn((X,Y), Old_Dir, M).
position(ID, (X,Y), Dir, T) :- position(ID, (X,Y), Dir, T-1), action(train(ID), wait, T-1).

%*% generate new position after action
% added cell((X,Y), _) <- cell@X,Y exists; to make sure we're not going out of bounds (shouldnt be needed in a viable environment)
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), n, T-1), not curve(X+1,Y), not symmetrical_switch(X+1,Y), not cell((X+1,Y),5633), not cell((X+1,Y),17411), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(train(ID), move_left, T-1), position(ID, (X+1,Y), e, T-1), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(train(ID), move_right, T-1), position(ID, (X+1,Y), w, T-1), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(train(ID), wait, T-1), position(ID, (X,Y), n, T-1).
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), e, T-1), not curve(X,Y-1), not symmetrical_switch(X,Y-1), not cell((X,Y-1),34864), not cell((X,Y-1),37408), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(train(ID), move_left, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(train(ID), move_right, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(train(ID), wait, T-1), position(ID, (X,Y), e, T-1).
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), s, T-1), not curve(X-1,Y), not symmetrical_switch(X-1,Y), not cell((X-1,Y),1097), not cell((X-1,Y),3089), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(train(ID), move_left, T-1), position(ID, (X-1,Y), w, T-1), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(train(ID), move_right, T-1), position(ID, (X-1,Y), e, T-1), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(train(ID), wait, T-1), position(ID, (X,Y), s, T-1).
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), w, T-1), not curve(X,Y+1), not symmetrical_switch(X,Y+1), not cell((X,Y+1),49186), not cell((X,Y+1),32872), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(train(ID), move_left, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(train(ID), move_right, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(train(ID), wait, T-1), position(ID, (X,Y), w, T-1).

% special case Type #1 curves
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y+1), 4608), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 4608), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y-1), 16386), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 16386), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y-1), 72), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 72), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y+1), 2064), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 2064), cell((X,Y), _).

% special case Type #6: symmetrical switches
% 20994
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 20994), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 20994), cell((X,Y), _).
% 2136
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 2136), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 2136), cell((X,Y), _).
% 16458
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y-1), 16458), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y-1), 16458), cell((X,Y), _).
% 6672
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y+1), 6672), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y+1), 6672), cell((X,Y), _).

% special case Type #2: simple switches
% 37408
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 37408), cell((X,Y), _).
% 17411
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y-1), 17411), cell((X,Y), _).
% 32872
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 32872), cell((X,Y), _).
% 3089
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y+1), 3089), cell((X,Y), _).
% 49186
position(ID, (X,Y), s, T) :- action(train(ID), move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 49186), cell((X,Y), _).
% 1097
position(ID, (X,Y), e, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y-1), 1097), cell((X,Y), _).
% 34864
position(ID, (X,Y), n, T) :- action(train(ID), move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 34864), cell((X,Y), _).
% 5633
position(ID, (X,Y), w, T) :- action(train(ID), move_forward, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y+1), 5633), cell((X,Y), _). *%


% possible movements on any track type depending on direction
% fact format: possible_movement(ID of track type, direction coming in, possible action).
% Type #1: straight tracks
possible_movement(32800, n, move_forward).
possible_movement(32800, s, move_forward).
possible_movement(32800, s, wait).

possible_movement(1025, e, move_forward).
possible_movement(1025, w, move_forward).

% Type #1: curves
possible_movement(4608, n, move_forward).
possible_movement(4608, e, move_forward).

possible_movement(16386, n, move_forward).
possible_movement(16386, w, move_forward).

possible_movement(72, s, move_forward).
possible_movement(72, w, move_forward).
possible_movement(72, s, wait).

possible_movement(2064, s, move_forward).
possible_movement(2064, e, move_forward).

% Type #2: simple switch left
possible_movement(37408, n, move_forward).
possible_movement(37408, n, move_left).
possible_movement(37408, e, move_forward).
possible_movement(37408, s, move_forward).

possible_movement(17411, w, move_forward).
possible_movement(17411, w, move_left).
possible_movement(17411, n, move_forward).
possible_movement(17411, e, move_forward).

possible_movement(32872, s, move_forward).
possible_movement(32872, s, move_left).
possible_movement(32872, n, move_forward).
possible_movement(32872, w, move_forward).
possible_movement(32872, s, wait).

possible_movement(3089, e, move_forward).
possible_movement(3089, e, move_left).
possible_movement(3089, s, move_forward).
possible_movement(3089, w, move_forward).

% Type #2: simple switch right
possible_movement(49186, n, move_forward).
possible_movement(49186, n, move_right).
possible_movement(49186, w, move_forward).
possible_movement(49186, s, move_forward).

possible_movement(1097, w, move_forward).
possible_movement(1097, w, move_right).
possible_movement(1097, s, move_forward).
possible_movement(1097, e, move_forward).

possible_movement(34864, s, move_forward).
possible_movement(34864, s, move_right).
possible_movement(34864, n, move_forward).
possible_movement(34864, e, move_forward).

possible_movement(5633, e, move_forward).
possible_movement(5633, e, move_right).
possible_movement(5633, n, move_forward).
possible_movement(5633, w, move_forward).

% Type #3: diamond crossing
possible_movement(33825, n, move_forward).
possible_movement(33825, e, move_forward).
possible_movement(33825, s, move_forward).
possible_movement(33825, w, move_forward).

% Type #4: single-slip switch
possible_movement(38433, n, move_forward).
possible_movement(38433, n, move_left).
possible_movement(38433, e, move_forward).
possible_movement(38433, e, move_right).
possible_movement(38433, s, move_forward).
possible_movement(38433, w, move_forward).

possible_movement(50211, n, move_forward).
possible_movement(50211, n, move_right).
possible_movement(50211, e, move_forward).
possible_movement(50211, s, move_forward).
possible_movement(50211, w, move_forward).
possible_movement(50211, w, move_left).

possible_movement(33897, n, move_forward).
possible_movement(33897, e, move_forward).
possible_movement(33897, s, move_forward).
possible_movement(33897, s, move_left).
possible_movement(33897, w, move_forward).
possible_movement(33897, w, move_right).

possible_movement(35889, n, move_forward).
possible_movement(35889, e, move_forward).
possible_movement(35889, e, move_left).
possible_movement(35889, s, move_forward).
possible_movement(35889, s, move_right).
possible_movement(35889, w, move_forward).

% Type #5: double-slip switch
possible_movement(38505, n, move_forward).
possible_movement(38505, n, move_left).
possible_movement(38505, e, move_forward).
possible_movement(38505, e, move_right).
possible_movement(38505, s, move_forward).
possible_movement(38505, s, move_left).
possible_movement(38505, w, move_forward).
possible_movement(38505, w, move_right).

possible_movement(52275, n, move_forward).
possible_movement(52275, n, move_right).
possible_movement(52275, e, move_forward).
possible_movement(52275, e, move_left).
possible_movement(52275, s, move_forward).
possible_movement(52275, s, move_right).
possible_movement(52275, w, move_forward).
possible_movement(52275, w, move_left).

% Type #6: symmetrical switch
possible_movement(20994, n, move_left).
possible_movement(20994, n, move_right).
possible_movement(20994, e, move_forward).
possible_movement(20994, w, move_forward).

possible_movement(16458, w, move_left).
possible_movement(16458, w, move_right).
possible_movement(16458, n, move_forward).
possible_movement(16458, s, move_forward).

possible_movement(2136, s, move_left).
possible_movement(2136, s, move_right).
possible_movement(2136, e, move_forward).
possible_movement(2136, w, move_forward).

possible_movement(6672, e, move_left).
possible_movement(6672, e, move_right).
possible_movement(6672, n, move_forward).
possible_movement(6672, s, move_forward).

% wait as possible_movement on any track type
possible_movement(TrackID, n, wait) :- cell(_, TrackID).
possible_movement(TrackID, e, wait) :- cell(_, TrackID).
possible_movement(TrackID, s, wait) :- cell(_, TrackID).
possible_movement(TrackID, w, wait) :- cell(_, TrackID).

% want a rule that choses exactly one of each possible movements per AS on the TrackID of the current position
{action(train(ID), Action, T) : possible_movement(TrackID, Dir, Action)}=1 :- 
    position(ID, (X,Y), Dir, T), cell((X,Y), TrackID), not end(ID, (X,Y), _), max_end_time(ID,E), T < E.

% minimize time
#minimize { T : end_time(ID, T) }.

% shortest path only
%:~ end_time(ID, T). [T@1]

% trains may not swap
:- position(ID1, (X1,Y1), _, T), position(ID2, (X2,Y2), _, T), position(ID1, (X2,Y2), _, T+1), position(ID2, (X1,Y1), _, T+1).

% no two trains on the same cell
:- position(ID1, (X,Y), Dir1, T), position(ID2, (X,Y), Dir2, T), ID1 != ID2.

% every train has to reach end
:- end(ID, (X,Y), _), not position(ID, (X,Y), _, _).

% trains have to be on time
:- end(ID, (X,Y), Arr), position(ID, (X,Y), _, T), T > Arr.


#show action/3.
%#show position/4.
%#show possible_movement/3.
