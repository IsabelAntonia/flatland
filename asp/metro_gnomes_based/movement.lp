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

% wait as possible_movement on any track type
possible_movement(TrackID, n, wait) :- cell((_,_), TrackID).
possible_movement(TrackID, e, wait) :- cell((_,_), TrackID).
possible_movement(TrackID, s, wait) :- cell((_,_), TrackID).
possible_movement(TrackID, w, wait) :- cell((_,_), TrackID).