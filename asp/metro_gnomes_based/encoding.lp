%--------------------------
% SPAWNING SETUP
%--------------------------
% Idle waiting
action(train(ID), wait, 0) :- train(ID).
action(train(ID), wait, 0..Dep-1) :- start(ID,(_,_),Dep,Dir), Dep > 0.

% Spawning action
action(train(ID), move_forward, Dep) :- start(ID, (X,Y), Dep, Dir), Dep > 0.
action(train(ID), move_forward, 1) :- start(ID, (X,Y), 0, Dir).

% Fact format: position(train ID, coordinates, facing direction, timestep)
% Init position
position(ID, (X,Y), Dir, Dep+1) :- start(ID, (X,Y), Dep, Dir), Dep >0.
position(ID, (X,Y), Dir, 2) :- start(ID, (X,Y), 0, Dir).

%--------------------------
% NEW FACT FORMATS
%--------------------------
% Latest arrival time of train
max_end_time(ID,E) :- end(ID,(_,_),E).

% Arrival time of train
end_time(ID, T) :- position(ID, (X,Y), _, T), end(ID, (X,Y), _).

% X-Y distance
% distance(ID,D) :- train(ID), start(ID, (X1,Y1), _, _), end(ID, (X2,Y2), _), D = |X2 - X1| + |Y2 - Y1|.
% prio(ID, P) :- train(ID), distance(ID,D), time_window(ID,TW), P = 1000*D/TW.

% Scheduled travel duration
scheduled_travel_duration(ID, TT) :- train(ID), start(ID, (_,_), Dep, _), end_time(ID, T), TT = T - Dep.

% Time window
time_window(ID,TW) :- train(ID), start(ID, (_,_), X, _), end(ID, (_,_), Y), TW = (Y+1) - X.

% Absolute slack
absolute_slack(ID, AS) :- train(ID), time_window(ID, TW), scheduled_travel_duration(ID, TD), AS = TW - TD.

% Travel duration to slack ratio
travel_to_slack_ratio(ID, TTSR) :- train(ID), absolute_slack(ID, AS), scheduled_travel_duration(ID, TD), TTSR = (1000*TD)/(AS+1).

%--------------------------
% PRIORITY ASSIGNMENT
%--------------------------
% Hardcoded for sacrifices.lp 
% prio(ID1,  3) :- train(ID1), ID1 = 0.
% prio(ID2, 1):- train(ID2), ID2 = 1.
% prio(ID3, 1) :- train(ID3), ID3 = 2.
% prio(ID4, 1) :- train(ID4), ID4 = 3.

% Hardcoded for sacrifices.lp
% prio(ID1,  3) :- train(ID1), ID1 = 0.
% prio(ID2, 2):- train(ID2), ID2 = 1.
% prio(ID3, 2) :- train(ID3), ID3 = 2.
% prio(ID4, 2) :- train(ID4), ID4 = 3.

% Hardcoded for situational_global.lp
% prio(ID1,  3) :- train(ID1), ID1 = 1.
% prio(ID2, 2):- train(ID2), ID2 = 2.
% prio(ID3, 1) :- train(ID3), ID3 = 3.

% Modulo encoding for evenly distributed priority assignment
% priority = weight for cost function
% priorityMap(0,3). % high
% priorityMap(1,2). % medium
% priorityMap(2,1). % low

% prio(ID, P) :- train(ID), R = ID \ 3, priorityMap(R,P).

%--------------------------
% POSITION GENERATION
%--------------------------
% Generate new position after action
% Restrictions in rule body instead of constraints later, to limit grounding of recursive rule
% Optimizations: only when new cell exists, new cell is a track, old cell allows movement
position(ID, (X+DX,Y+DY), New_Dir, T) :- position(ID, (X,Y), Old_Dir, T-1), action(train(ID), M, T-1), movement(M, Old_Dir, (DX,DY), New_Dir), 
                                        M != wait, not forced_turn((X,Y), Old_Dir,_), cell((X+DX,Y+DY),TrackID), TrackID != 0,
                                        cell((X,Y), TrackID_old), possible_movement(TrackID_old, Old_Dir, M).

position(ID, (X+DX,Y+DY), New_Dir, T) :- position(ID, (X,Y), Old_Dir, T-1), action(train(ID), move_forward, T-1), movement(M, Old_Dir, (DX,DY), New_Dir), 
                                        forced_turn((X,Y), Old_Dir, M), cell((X+DX,Y+DY),TrackID), TrackID != 0.

position(ID, (X,Y), Dir, T) :- position(ID, (X,Y), Dir, T-1), action(train(ID), wait, T-1).

%--------------------------
% ACTION GENERATION CHOICE RULE
%--------------------------
% A rule that choses exactly one of each possible movements per AS on the TrackID of the current position
{action(train(ID), Action, T) : possible_movement(TrackID, Dir, Action)}=1 :- 
    position(ID, (X,Y), Dir, T), cell((X,Y), TrackID), not end(ID, (X,Y), _), max_end_time(ID,E), T < (E+1).    

% LESS COMPLEX choice rule DID NOT result in performance increase                      
%{action(train(ID), Action, T) : possible_movement(TrackID, Dir, Action)}=1 :- 
%                    position(ID, (X,Y), Dir, T), cell((X,Y), TrackID), max_end_time(ID,E), T < (E+1).                                     

% No more action generation after reaching end for less complex choice rule
%:- action(train(ID),_,T), position(ID, (X,Y), _, T), end(ID, (X,Y), _).

% No more action generation after deadline for less complex choice rule
% this results in a lot more grounding and DOES NOT FINISH
%:- action(train(ID),_,T), max_end_time(ID,E), T > E.

%--------------------------
% CONSTRAINTS
%--------------------------
% Trains may not swap
:- position(ID1, (X1,Y1), _, T), position(ID2, (X2,Y2), _, T), position(ID1, (X2,Y2), _, T+1), position(ID2, (X1,Y1), _, T+1), ID1 != ID2.

% No two trains on the same cell
:- position(ID1, (X,Y), Dir1, T), position(ID2, (X,Y), Dir2, T), ID1 != ID2.

% Every train has to reach destination
:- end(ID, (X,Y), _), not position(ID, (X,Y), _, _).

%--------------------------
% HARD ENFORCING PRIORITIES
%--------------------------
% #minimize { T3@1 : end_time(ID3, T3), prio(ID3, low)}.
% #minimize { T2@2 : end_time(ID2, T2), prio(ID2, medium)}.
% #minimize { T1@3 : end_time(ID1, T1), prio(ID1, high)}.

%--------------------------
% COST FUNCTION EXPLICIT PRIORITIES
%--------------------------
%#minimize { P*T, ID : prio(ID, P), end_time(ID, T)}.

%--------------------------
% COST FUNCTION ROBUSTNESS
%--------------------------
#minimize {TTSR, ID: travel_to_slack_ratio(ID, TTSR)}.

%--------------------------
% OUTPUT
%--------------------------
%#show time_window/2.
%#show distance/2.
%#show prio/2.
%#show urgency/2.
%#show scheduled_travel_duration/2.
%#show time_window/2.
#show action/3.
%#show end_time/2.
%#show position/4.
%#show possible_movement/3.
