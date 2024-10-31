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
end(1,(1,2),20).
%end(1,(2,0),20).
% init position
position(ID, (X,Y), Dir, 1) :- start(ID, (X,Y), Dep, Dir).

% For all Track Types
%position(ID, (X,Y), Dir, T) :- cell((X,Y),TT), position(ID, (X,Y), Dir, T-1).

% Track Type 4608
{
position(ID, (X,Y-1), w, T): cell((X,Y),4608), position(ID, (X,Y), n, T-1);
position(ID, (X+1,Y), s, T) : cell((X,Y),4608), position(ID, (X,Y), e, T-1)
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).


% Track Type 16386
{
position(ID, (X,Y+1), e, T) : cell((X,Y),16386), position(ID, (X,Y), n, T-1);
position(ID, (X+1,Y), s, T) : cell((X,Y),16386), position(ID, (X,Y), w, T-1)
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).


% Track Type 72
{
position(ID, (X,Y+1), e, T) : cell((X,Y),72), position(ID, (X,Y), s, T-1);
position(ID, (X-1,Y), n, T) : cell((X,Y),72), position(ID, (X,Y), w, T-1)
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).

% Track Type 2064
{
position(ID, (X,Y-1), w, T) : cell((X,Y),2064), position(ID, (X,Y), s, T-1);
position(ID, (X-1,Y), n, T) : cell((X,Y),2064), position(ID, (X,Y), e, T-1)
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).


% Track Type 20994
{
position(ID, (X,Y-1), w, T) : cell((X,Y),20994), position(ID, (X,Y), n, T-1);
position(ID, (X,Y+1), e, T) : cell((X,Y),20994), position(ID, (X,Y), n, T-1);
position(ID, (X+1,Y), s, T) : cell((X,Y),20994), position(ID, (X,Y), e, T-1);
position(ID, (X+1,Y), s, T) : cell((X,Y),20994), position(ID, (X,Y), w, T-1)
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).

% Track Type 2136
{position(ID, (X,Y-1), w, T), position(ID, (X,Y+1), e, T)}=1 :- cell((X,Y),2136), position(ID, (X,Y), s, T-1).

{
%position(ID, (X,Y-1), w, T) : cell((X,Y),2136), position(ID, (X,Y), s, T-1);
%position(ID, (X,Y+1), e, T) : cell((X,Y),2136), position(ID, (X,Y), s, T-1);
position(ID, (X-1,Y), n, T) : cell((X,Y),2136), position(ID, (X,Y), e, T-1);
position(ID, (X-1,Y), n, T) : cell((X,Y),2136), position(ID, (X,Y), w, T-1)
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).


% Track Type 32800
{
position(ID, (X-1,Y), n, T) : cell((X,Y),32800), position(ID, (X,Y), n, T-1);
position(ID, (X+1,Y), s, T) : cell((X,Y),32800), position(ID, (X,Y), s, T-1) 
} :- position(ID, (X,Y), _, _), not end(ID, (X,Y), _).

%#show time/1.
#show position/4.

% every train has to reach end
:- end(ID, (X,Y), _), not position(ID, (X,Y), _, _).


% Trains may not swap

% Trains may not occupy the same spot

% Minimize Time Steps for that have to count time steps 