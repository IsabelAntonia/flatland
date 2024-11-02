%* cell((0,0), 16386). cell((0,1), 5633). cell((0,2), 1025). 
cell((1,0), 72). cell((1,1), 2064). cell((1,2), 0). *%


% environment0
cell((0,0),16386).
cell((0,1),20994).
cell((0,2),4608).
cell((1,0),32800).
cell((1,1),32800).
cell((1,2),32800).
cell((2,0),72).
cell((2,1),2136).
cell((2,2),2064).

train(1).
start(1,(1,0),0,n).
end(1,(1,2),10).

train(2).
start(2,(1,2),0,e).
end(2,(1,0),10).

max_end_time(ID,E) :- end(ID,(_,_),E).
% init position
% fact format: position(train ID, coordinates, facing direction, timestep)
position(ID, (X,Y), Dir, 0) :- start(ID, (X,Y), Dep, Dir).
curve(X,Y) :- cell((X,Y),72).
curve(X,Y) :- cell((X,Y),4608).
curve(X,Y) :- cell((X,Y),16386).
curve(X,Y) :- cell((X,Y),2064).
special_track(X,Y) :- cell((X,Y),20994).

%step_count(ID, N) :- train(ID),N = * { T : position(ID, (_,_), _, T)}.
end_time(ID, T) :- position(ID, (X,Y), _, T), end(ID, (X,Y), _).

% generate new position after action
% added cell((X,Y), _) <- cell@X,Y exists; to make sure we're not going out of bounds (shouldnt be needed in a viable environment)
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), n, T-1), not curve(X+1,Y), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_left, T-1), position(ID, (X+1,Y), e, T-1), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_right, T-1), position(ID, (X+1,Y), w, T-1), cell((X,Y), _).
%position(ID, (X,Y), n, T) :- action(ID, wait, T-1), position(ID, (X,Y), n, T-1).
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), e, T-1), not curve(X,Y-1), not special_track(X,Y-1), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_left, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_right, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y), _).
%position(ID, (X,Y), e, T) :- action(ID, wait, T-1), position(ID, (X,Y), e, T-1).
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), s, T-1), not curve(X-1,Y), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_left, T-1), position(ID, (X-1,Y), w, T-1), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_right, T-1), position(ID, (X-1,Y), e, T-1), cell((X,Y), _).
%position(ID, (X,Y), s, T) :- action(ID, wait, T-1), position(ID, (X,Y), s, T-1).
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), w, T-1), not curve(X,Y+1), not special_track(X,Y+1),  cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_left, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_right, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y), _).
%position(ID, (X,Y), w, T) :- action(ID, wait, T-1), position(ID, (X,Y), w, T-1).

% special case Type #1 curves
% TODO: add other cases with curves and move_forward (switches from behind)
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y+1), 4608), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 4608), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y-1), 16386), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 16386), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y-1), 72), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 72), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y+1), 2064), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 2064), cell((X,Y), _).
% symmetrical switches
% 20994
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 20994), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 20994), cell((X,Y), _).
% 2136
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 2136), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 2136), cell((X,Y), _).
% 16458
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y-1), 16458), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y-1), 16458), cell((X,Y), _).
% 6672
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y+1), 6672), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y+1), 6672), cell((X,Y), _).

% 8 Simple Switch


% Track Type 2 ToDO

% possible movements on any track type depending on direction
% fact format: possible_movement(ID of track type, direction coming in, possible action).
% Type #1: straight tracks
possible_movement(32800, n, move_forward).
possible_movement(32800, s, move_forward).

possible_movement(1025, e, move_forward).
possible_movement(1025, w, move_forward).

% Type #1: curves
possible_movement(4608, n, move_forward).
possible_movement(4608, e, move_forward).

possible_movement(16386, n, move_forward).
possible_movement(16386, w, move_forward).

possible_movement(72, s, move_forward).
possible_movement(72, w, move_forward).

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

{action(ID, Action, T) : possible_movement(TrackID, Dir, Action)}=1 :- position(ID, (X,Y), Dir, T), cell((X,Y), TrackID), not end(ID, (X,Y), _), max_end_time(ID,E), T < E.

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
%:- end(ID, (X,Y), Arr), position(ID, (X,Y), _, T), T > Arr.


#show action/3.
%#show position/4.