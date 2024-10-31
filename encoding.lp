
cell((0,0), 16386). cell((0,1), 5633). cell((0,2), 1025). 
cell((1,0), 72). cell((1,1), 2064). cell((1,2), 0).

%*
cell((0,0),16386).
cell((0,1),20994).
cell((0,2),4608).
cell((1,0),32800).
cell((1,1),32800).
cell((1,2),32800).
cell((2,0),72).
%cell((2,1),2136).
cell((2,2),2064). *%

train(1).
start(1,(0,0),0,n).
end(1,(0,2),20).
max_end_time(ID,E) :- end(ID,(_,_),E).
% init position
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
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), s, T-1), not curve(X+1,Y), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_left, T-1), position(ID, (X-1,Y), w, T-1), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_right, T-1), position(ID, (X-1,Y), e, T-1), cell((X,Y), _).
%position(ID, (X,Y), s, T) :- action(ID, wait, T-1), position(ID, (X,Y), s, T-1).
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), w, T-1), not curve(X+1,Y), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_left, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_right, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y), _).
%position(ID, (X,Y), w, T) :- action(ID, wait, T-1), position(ID, (X,Y), w, T-1).

% special case Type #1 curves
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), n, T-1), cell((X,Y+1), 4608), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 4608), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), n, T-1), cell((X,Y-1), 16386), cell((X,Y), _).
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 16386), cell((X,Y), _).
position(ID, (X,Y), e, T) :- action(ID, move_forward, T-1), position(ID, (X,Y-1), s, T-1), cell((X,Y-1), 72), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 72), cell((X,Y), _).
position(ID, (X,Y), w, T) :- action(ID, move_forward, T-1), position(ID, (X,Y+1), s, T-1), cell((X,Y+1), 2064), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 2064), cell((X,Y), _).
% new for 20994
position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), e, T-1), cell((X-1,Y), 20994), cell((X,Y), _).
%position(ID, (X,Y), s, T) :- action(ID, move_forward, T-1), position(ID, (X-1,Y), w, T-1), cell((X-1,Y), 20994), cell((X,Y), _).
% new for 2136
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), e, T-1), cell((X+1,Y), 2136), cell((X,Y), _).
position(ID, (X,Y), n, T) :- action(ID, move_forward, T-1), position(ID, (X+1,Y), w, T-1), cell((X+1,Y), 2136), cell((X,Y), _).


% Track Type 2 ToDO

% possible movements on any track type depending on direction
% Type #1: straight tracks
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 32800).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 32800).

possible_movement((X,Y), e, move_forward) :- cell((X,Y), 1025).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 1025).

% Type #1: curves
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 4608).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 4608).

possible_movement((X,Y), n, move_forward) :- cell((X,Y), 16386).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 16386).

possible_movement((X,Y), s, move_forward) :- cell((X,Y), 72).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 72).

possible_movement((X,Y), s, move_forward) :- cell((X,Y), 2064).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 2064).

% Type #2: simple switch left
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 37408).
possible_movement((X,Y), n, move_left) :- cell((X,Y), 37408).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 37408).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 37408).

possible_movement((X,Y), w, move_forward) :- cell((X,Y), 17411).
possible_movement((X,Y), w, move_left) :- cell((X,Y), 17411).
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 17411).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 17411).

possible_movement((X,Y), s, move_forward) :- cell((X,Y), 32872).
possible_movement((X,Y), s, move_left) :- cell((X,Y), 32872).
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 32872).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 32872).

possible_movement((X,Y), e, move_forward) :- cell((X,Y), 3089).
possible_movement((X,Y), e, move_left) :- cell((X,Y), 3089).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 3089).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 3089).

% Type #2: simple switch right
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 49186).
possible_movement((X,Y), n, move_right) :- cell((X,Y), 49186).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 49186).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 49186).

possible_movement((X,Y), w, move_forward) :- cell((X,Y), 1097).
possible_movement((X,Y), w, move_right) :- cell((X,Y), 1097).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 1097).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 1097).

possible_movement((X,Y), s, move_forward) :- cell((X,Y), 34864).
possible_movement((X,Y), s, move_right) :- cell((X,Y), 34864).
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 34864).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 34864).

possible_movement((X,Y), e, move_forward) :- cell((X,Y), 5633).
possible_movement((X,Y), e, move_right) :- cell((X,Y), 5633).
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 5633).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 5633).

% Type #3: diamond crossing
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 33825).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 33825).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 33825).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 33825).

% Type #4: single-slip switch
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 38433).
possible_movement((X,Y), n, move_left) :- cell((X,Y), 38433).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 38433).
possible_movement((X,Y), e, move_right) :- cell((X,Y), 38433).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 38433).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 38433).

possible_movement((X,Y), n, move_forward) :- cell((X,Y), 50211).
possible_movement((X,Y), n, move_right) :- cell((X,Y), 50211).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 50211).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 50211).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 50211).
possible_movement((X,Y), w, move_left) :- cell((X,Y), 50211).

possible_movement((X,Y), n, move_forward) :- cell((X,Y), 33897).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 33897).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 33897).
possible_movement((X,Y), s, move_left) :- cell((X,Y), 33897).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 33897).
possible_movement((X,Y), w, move_right) :- cell((X,Y), 33897).

possible_movement((X,Y), n, move_forward) :- cell((X,Y), 35889).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 35889).
possible_movement((X,Y), e, move_left) :- cell((X,Y), 35889).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 35889).
possible_movement((X,Y), s, move_right) :- cell((X,Y), 35889).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 35889).

% Type #5: double-slip switch
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 38505).
possible_movement((X,Y), n, move_left) :- cell((X,Y), 38505).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 38505).
possible_movement((X,Y), e, move_right) :- cell((X,Y), 38505).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 38505).
possible_movement((X,Y), s, move_left) :- cell((X,Y), 38505).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 38505).
possible_movement((X,Y), w, move_right) :- cell((X,Y), 38505).

possible_movement((X,Y), n, move_forward) :- cell((X,Y), 52275).
possible_movement((X,Y), n, move_right) :- cell((X,Y), 52275).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 52275).
possible_movement((X,Y), e, move_left) :- cell((X,Y), 52275).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 52275).
possible_movement((X,Y), s, move_right) :- cell((X,Y), 52275).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 52275).
possible_movement((X,Y), w, move_left) :- cell((X,Y), 52275).

% Type #6: symmetrical switch
possible_movement((X,Y), n, move_left) :- cell((X,Y), 20994).
possible_movement((X,Y), n, move_right) :- cell((X,Y), 20994).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 20994).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 20994).

possible_movement((X,Y), w, move_left) :- cell((X,Y), 16458).
possible_movement((X,Y), w, move_right) :- cell((X,Y), 16458).
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 16458).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 16458).

possible_movement((X,Y), s, move_left) :- cell((X,Y), 2136).
possible_movement((X,Y), s, move_right) :- cell((X,Y), 2136).
possible_movement((X,Y), e, move_forward) :- cell((X,Y), 2136).
possible_movement((X,Y), w, move_forward) :- cell((X,Y), 2136).

possible_movement((X,Y), e, move_left) :- cell((X,Y), 6672).
possible_movement((X,Y), e, move_right) :- cell((X,Y), 6672).
possible_movement((X,Y), n, move_forward) :- cell((X,Y), 6672).
possible_movement((X,Y), s, move_forward) :- cell((X,Y), 6672).

{action(ID, M, T) : possible_movement((X,Y), Dir, M)}=1 :- position(ID, (X,Y), Dir, T), not end(ID, (X,Y), _), max_end_time(ID,E), T < E.

% minimize time
#minimize { T : end_time(ID, T) }.

% shortest path only
:~ end_time(ID, T). [T@1]

% trains may not swap
%:- position(ID1, (X1,Y1), _, T), position(ID2, (X2,Y2), _, T), position(ID1, (X2,Y2), _, T+1), position(ID2, (X1,Y1), _, T+1).

% no two trains on the same cell
%:- position(ID1, (X,Y), Dir1, T), position(ID2, (X,Y), Dir2, T), ID1 != ID2.

% every train has to reach end
:- end(ID, (X,Y), _), not position(ID, (X,Y), _, _).

% trains have to be in time
%:- end(ID, (X,Y), Arr), position(ID, (X,Y), _, T), T > Arr.


#show action/3.
#show position/4.